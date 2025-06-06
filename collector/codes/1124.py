import os
import sys
from functools import partial

import numpy as np
import shm_util as su
import test_util as tu
import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import *

if sys.version_info >= (3, 0):
    import queue
else:
    import Queue as queue


if sys.version_info.major == 3:
    unicode = bytes

_seen_request_ids = set()


_tritonserver_ipaddr = os.environ.get("TRITONSERVER_IPADDR", "localhost")


def _unique_request_id():
    if len(_seen_request_ids) == 0:
        return 1
    else:
        return max(_seen_request_ids) + 1


def _range_repr_dtype(dtype):
    if dtype == np.float64:
        return np.int32
    elif dtype == np.float32:
        return np.int16
    elif dtype == np.float16:
        return np.int8
    elif dtype == np.object_:
        return np.int32
    return dtype


def serialize_byte_tensor_list(tensor_values):
    tensor_list = []
    for tensor_value in tensor_values:
        tensor_list.append(serialize_byte_tensor(tensor_value))
    return tensor_list


class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()


def completion_callback(user_data, result, error):

    user_data._completed_requests.put((result, error))


def infer_exact(
    tester,
    pf,
    tensor_shape,
    batch_size,
    input_dtype,
    output0_dtype,
    output1_dtype,
    output0_raw=True,
    output1_raw=True,
    model_version=None,
    swap=False,
    outputs=("OUTPUT0", "OUTPUT1"),
    use_http=True,
    use_grpc=True,
    use_http_json_tensors=True,
    skip_request_id_check=False,
    use_streaming=True,
    correlation_id=0,
    shm_region_names=None,
    precreated_shm_regions=None,
    use_system_shared_memory=False,
    use_cuda_shared_memory=False,
    priority=0,
    network_timeout=60.0,
):

    if use_system_shared_memory:
        import tritonclient.utils.shared_memory as shm
    if use_cuda_shared_memory:
        import tritonclient.utils.cuda_shared_memory as cudashm

    tester.assertTrue(use_http or use_grpc or use_streaming)

    configs = []
    if use_http:
        configs.append((f"{_tritonserver_ipaddr}:8000", "http", False, True))
        if output0_raw == output1_raw:

            if (
                use_http_json_tensors
                and (input_dtype != np.float16)
                and (output0_dtype != np.float16)
                and (output1_dtype != np.float16)
            ):
                configs.append((f"{_tritonserver_ipaddr}:8000", "http", False, False))
    if use_grpc:
        configs.append((f"{_tritonserver_ipaddr}:8001", "grpc", False, False))
    if use_streaming:
        configs.append((f"{_tritonserver_ipaddr}:8001", "grpc", True, False))

    rinput_dtype = _range_repr_dtype(input_dtype)
    routput0_dtype = _range_repr_dtype(output0_dtype if output0_raw else np.float32)
    routput1_dtype = _range_repr_dtype(output1_dtype if output1_raw else np.float32)
    val_min = (
        max(
            np.iinfo(rinput_dtype).min,
            np.iinfo(routput0_dtype).min,
            np.iinfo(routput1_dtype).min,
        )
        / 2
    )
    val_max = (
        min(
            np.iinfo(rinput_dtype).max,
            np.iinfo(routput0_dtype).max,
            np.iinfo(routput1_dtype).max,
        )
        / 2
    )

    input0_array = np.random.randint(
        low=val_min, high=val_max, size=tensor_shape, dtype=rinput_dtype
    )
    input1_array = np.random.randint(
        low=val_min, high=val_max, size=tensor_shape, dtype=rinput_dtype
    )
    if input_dtype != np.object_:
        input0_array = input0_array.astype(input_dtype)
        input1_array = input1_array.astype(input_dtype)

    if val_min == 0:

        tmp = np.where(input0_array < input1_array, input1_array, input0_array)
        input1_array = np.where(input0_array < input1_array, input0_array, input1_array)
        input0_array = tmp

    if not swap:
        output0_array = input0_array + input1_array
        output1_array = input0_array - input1_array
    else:
        output0_array = input0_array - input1_array
        output1_array = input0_array + input1_array

    if output0_dtype == np.object_:
        output0_array = np.array(
            [unicode(str(x), encoding="utf-8") for x in (output0_array.flatten())],
            dtype=object,
        ).reshape(output0_array.shape)
    else:
        output0_array = output0_array.astype(output0_dtype)
    if output1_dtype == np.object_:
        output1_array = np.array(
            [unicode(str(x), encoding="utf-8") for x in (output1_array.flatten())],
            dtype=object,
        ).reshape(output1_array.shape)
    else:
        output1_array = output1_array.astype(output1_dtype)

    if input_dtype == np.object_:
        in0n = np.array(
            [str(x) for x in input0_array.reshape(input0_array.size)], dtype=object
        )
        input0_array = in0n.reshape(input0_array.shape)
        in1n = np.array(
            [str(x) for x in input1_array.reshape(input1_array.size)], dtype=object
        )
        input1_array = in1n.reshape(input1_array.shape)

    if output0_dtype == np.object_:
        if batch_size == 1:
            output0_array_tmp = serialize_byte_tensor_list([output0_array])
        else:
            output0_array_tmp = serialize_byte_tensor_list(output0_array)
    else:
        output0_array_tmp = output0_array

    if output1_dtype == np.object_:
        if batch_size == 1:
            output1_array_tmp = serialize_byte_tensor_list([output1_array])
        else:
            output1_array_tmp = serialize_byte_tensor_list(output1_array)
    else:
        output1_array_tmp = output1_array

    if output0_dtype == np.object_:
        output0_byte_size = sum([serialized_byte_size(o0) for o0 in output0_array_tmp])
    else:
        output0_byte_size = sum([o0.nbytes for o0 in output0_array_tmp])

    if output1_dtype == np.object_:
        output1_byte_size = sum([serialized_byte_size(o1) for o1 in output1_array_tmp])
    else:
        output1_byte_size = sum([o1.nbytes for o1 in output1_array_tmp])

    if batch_size == 1:
        input0_list = [input0_array]
        input1_list = [input1_array]
    else:
        input0_list = [x for x in input0_array]
        input1_list = [x for x in input1_array]

    if input_dtype == np.object_:
        input0_list_tmp = serialize_byte_tensor_list(input0_list)
        input1_list_tmp = serialize_byte_tensor_list(input1_list)
    else:
        input0_list_tmp = input0_list
        input1_list_tmp = input1_list

    if input_dtype == np.object_:
        input0_byte_size = sum([serialized_byte_size(i0) for i0 in input0_list_tmp])
        input1_byte_size = sum([serialized_byte_size(i1) for i1 in input1_list_tmp])
    else:
        input0_byte_size = sum([i0.nbytes for i0 in input0_list_tmp])
        input1_byte_size = sum([i1.nbytes for i1 in input1_list_tmp])

    if model_version is not None:
        model_version = str(model_version)
    else:
        model_version = ""

    inferAndCheckResults(
        tester,
        configs,
        pf,
        batch_size,
        model_version,
        input_dtype,
        output0_dtype,
        output1_dtype,
        tensor_shape,
        input0_array,
        input1_array,
        output0_array,
        output1_array,
        output0_raw,
        output1_raw,
        outputs,
        precreated_shm_regions,
        input0_list_tmp,
        input1_list_tmp,
        shm_region_names,
        input0_byte_size,
        input1_byte_size,
        output0_byte_size,
        output1_byte_size,
        use_system_shared_memory,
        use_cuda_shared_memory,
        network_timeout,
        skip_request_id_check,
    )


def inferAndCheckResults(
    tester,
    configs,
    pf,
    batch_size,
    model_version,
    input_dtype,
    output0_dtype,
    output1_dtype,
    tensor_shape,
    input0_array,
    input1_array,
    output0_array,
    output1_array,
    output0_raw,
    output1_raw,
    outputs,
    precreated_shm_regions,
    input0_list_tmp,
    input1_list_tmp,
    shm_region_names,
    input0_byte_size,
    input1_byte_size,
    output0_byte_size,
    output1_byte_size,
    use_system_shared_memory,
    use_cuda_shared_memory,
    network_timeout,
    skip_request_id_check,
):

    if use_system_shared_memory:
        import tritonclient.utils.shared_memory as shm
    if use_cuda_shared_memory:
        import tritonclient.utils.cuda_shared_memory as cudashm
    num_classes = 3

    model_name = tu.get_model_name(pf, input_dtype, output0_dtype, output1_dtype)
    if configs[0][1] == "http":
        metadata_client = httpclient.InferenceServerClient(configs[0][0], verbose=True)
        metadata = metadata_client.get_model_metadata(model_name)
        platform = metadata["platform"]
    else:
        metadata_client = grpcclient.InferenceServerClient(configs[0][0], verbose=True)
        metadata = metadata_client.get_model_metadata(model_name)
        platform = metadata.platform

    INPUT0 = "INPUT0"
    INPUT1 = "INPUT1"

    if platform == "pytorch_libtorch":
        OUTPUT0 = "OUTPUT__0"
        OUTPUT1 = "OUTPUT__1"
    else:
        OUTPUT0 = "OUTPUT0"
        OUTPUT1 = "OUTPUT1"

    shm_regions, shm_handles = su.create_set_shm_regions(
        input0_list_tmp,
        input1_list_tmp,
        output0_byte_size,
        output1_byte_size,
        outputs,
        shm_region_names,
        precreated_shm_regions,
        use_system_shared_memory,
        use_cuda_shared_memory,
    )
    try:
        for config in configs:
            model_name = tu.get_model_name(
                pf, input_dtype, output0_dtype, output1_dtype
            )

            if config[1] == "http":
                triton_client = httpclient.InferenceServerClient(
                    config[0], verbose=True, network_timeout=network_timeout
                )
            else:
                triton_client = grpcclient.InferenceServerClient(
                    config[0], verbose=True
                )

            inputs = []
            if config[1] == "http":
                inputs.append(
                    httpclient.InferInput(
                        INPUT0, tensor_shape, np_to_triton_dtype(input_dtype)
                    )
                )
                inputs.append(
                    httpclient.InferInput(
                        INPUT1, tensor_shape, np_to_triton_dtype(input_dtype)
                    )
                )
            else:
                inputs.append(
                    grpcclient.InferInput(
                        INPUT0, tensor_shape, np_to_triton_dtype(input_dtype)
                    )
                )
                inputs.append(
                    grpcclient.InferInput(
                        INPUT1, tensor_shape, np_to_triton_dtype(input_dtype)
                    )
                )

            if not (use_cuda_shared_memory or use_system_shared_memory):
                if config[1] == "http":
                    inputs[0].set_data_from_numpy(input0_array, binary_data=config[3])
                    inputs[1].set_data_from_numpy(input1_array, binary_data=config[3])
                else:
                    inputs[0].set_data_from_numpy(input0_array)
                    inputs[1].set_data_from_numpy(input1_array)
            else:

                su.register_add_shm_regions(
                    inputs,
                    outputs,
                    shm_regions,
                    precreated_shm_regions,
                    shm_handles,
                    input0_byte_size,
                    input1_byte_size,
                    output0_byte_size,
                    output1_byte_size,
                    use_system_shared_memory,
                    use_cuda_shared_memory,
                    triton_client,
                )

            if batch_size == 1:
                expected0_sort_idx = [
                    np.flip(np.argsort(x.flatten()), 0)
                    for x in output0_array.reshape((1,) + tensor_shape)
                ]
                expected1_sort_idx = [
                    np.flip(np.argsort(x.flatten()), 0)
                    for x in output1_array.reshape((1,) + tensor_shape)
                ]
            else:
                expected0_sort_idx = [
                    np.flip(np.argsort(x.flatten()), 0)
                    for x in output0_array.reshape(tensor_shape)
                ]
                expected1_sort_idx = [
                    np.flip(np.argsort(x.flatten()), 0)
                    for x in output1_array.reshape(tensor_shape)
                ]

            output_req = []
            i = 0
            if "OUTPUT0" in outputs:
                if len(shm_regions) != 0:
                    if config[1] == "http":
                        output_req.append(
                            httpclient.InferRequestedOutput(
                                OUTPUT0, binary_data=config[3]
                            )
                        )
                    else:
                        output_req.append(grpcclient.InferRequestedOutput(OUTPUT0))

                    output_req[-1].set_shared_memory(
                        shm_regions[2] + "_data", output0_byte_size
                    )
                else:
                    if output0_raw:
                        if config[1] == "http":
                            output_req.append(
                                httpclient.InferRequestedOutput(
                                    OUTPUT0, binary_data=config[3]
                                )
                            )
                        else:
                            output_req.append(grpcclient.InferRequestedOutput(OUTPUT0))
                    else:
                        if config[1] == "http":
                            output_req.append(
                                httpclient.InferRequestedOutput(
                                    OUTPUT0,
                                    binary_data=config[3],
                                    class_count=num_classes,
                                )
                            )
                        else:
                            output_req.append(
                                grpcclient.InferRequestedOutput(
                                    OUTPUT0, class_count=num_classes
                                )
                            )
                i += 1
            if "OUTPUT1" in outputs:
                if len(shm_regions) != 0:
                    if config[1] == "http":
                        output_req.append(
                            httpclient.InferRequestedOutput(
                                OUTPUT1, binary_data=config[3]
                            )
                        )
                    else:
                        output_req.append(grpcclient.InferRequestedOutput(OUTPUT1))

                    output_req[-1].set_shared_memory(
                        shm_regions[2 + i] + "_data", output1_byte_size
                    )
                else:
                    if output1_raw:
                        if config[1] == "http":
                            output_req.append(
                                httpclient.InferRequestedOutput(
                                    OUTPUT1, binary_data=config[3]
                                )
                            )
                        else:
                            output_req.append(grpcclient.InferRequestedOutput(OUTPUT1))
                    else:
                        if config[1] == "http":
                            output_req.append(
                                httpclient.InferRequestedOutput(
                                    OUTPUT1,
                                    binary_data=config[3],
                                    class_count=num_classes,
                                )
                            )
                        else:
                            output_req.append(
                                grpcclient.InferRequestedOutput(
                                    OUTPUT1, class_count=num_classes
                                )
                            )

            if config[2]:
                user_data = UserData()
                triton_client.start_stream(partial(completion_callback, user_data))
                try:
                    results = triton_client.async_stream_infer(
                        model_name,
                        inputs,
                        model_version=model_version,
                        outputs=output_req,
                        request_id=str(_unique_request_id()),
                    )
                except Exception as e:
                    triton_client.stop_stream()
                    raise e
                triton_client.stop_stream()
                (results, error) = user_data._completed_requests.get()
                if error is not None:
                    raise error
            else:
                results = triton_client.infer(
                    model_name,
                    inputs,
                    model_version=model_version,
                    outputs=output_req,
                    request_id=str(_unique_request_id()),
                )

            last_response = results.get_response()

            if not skip_request_id_check:
                global _seen_request_ids
                if config[1] == "http":
                    request_id = int(last_response["id"])
                else:
                    request_id = int(last_response.id)
                tester.assertFalse(
                    request_id in _seen_request_ids, "request_id: {}".format(request_id)
                )
                _seen_request_ids.add(request_id)

            if config[1] == "http":
                response_model_name = last_response["model_name"]
                if model_version != "":
                    response_model_version = last_response["model_version"]
                response_outputs = last_response["outputs"]
            else:
                response_model_name = last_response.model_name
                if model_version != "":
                    response_model_version = last_response.model_version
                response_outputs = last_response.outputs

            tester.assertEqual(response_model_name, model_name)

            if model_version != "":
                tester.assertEqual(str(response_model_version), model_version)

            tester.assertEqual(len(response_outputs), len(outputs))

            for result in response_outputs:
                if config[1] == "http":
                    result_name = result["name"]
                else:
                    result_name = result.name

                if (result_name == OUTPUT0 and output0_raw) or (
                    result_name == OUTPUT1 and output1_raw
                ):
                    if use_system_shared_memory or use_cuda_shared_memory:
                        if result_name == OUTPUT0:
                            shm_handle = shm_handles[2]
                        else:
                            shm_handle = shm_handles[3]

                        output = results.get_output(result_name)
                        if config[1] == "http":
                            output_datatype = output["datatype"]
                            output_shape = output["shape"]
                        else:
                            output_datatype = output.datatype
                            output_shape = output.shape
                        output_dtype = triton_to_np_dtype(output_datatype)
                    if use_system_shared_memory:
                        output_data = shm.get_contents_as_numpy(
                            shm_handle, output_dtype, output_shape
                        )
                    elif use_cuda_shared_memory:
                        output_data = cudashm.get_contents_as_numpy(
                            shm_handle, output_dtype, output_shape
                        )
                    else:
                        output_data = results.as_numpy(result_name)
                        if (output_data.dtype == np.object_) and (not config[3]):
                            if config[1] == "http":
                                output_data = np.array(
                                    [
                                        unicode(str(x), encoding="utf-8")
                                        for x in (output_data.flatten())
                                    ],
                                    dtype=np.object_,
                                ).reshape(output_data.shape)
                            elif config[1] == "grpc":
                                output_data = np.array(
                                    [x for x in (output_data.flatten())],
                                    dtype=np.object_,
                                ).reshape(output_data.shape)

                    if result_name == OUTPUT0:
                        tester.assertTrue(
                            np.array_equal(output_data, output0_array),
                            "{}, {} expected: {}, got {}".format(
                                model_name, OUTPUT0, output0_array, output_data
                            ),
                        )
                    elif result_name == OUTPUT1:
                        tester.assertTrue(
                            np.array_equal(output_data, output1_array),
                            "{}, {} expected: {}, got {}".format(
                                model_name, OUTPUT1, output1_array, output_data
                            ),
                        )
                    else:
                        tester.assertTrue(
                            False, "unexpected raw result {}".format(result_name)
                        )
                else:
                    for b in range(batch_size):

                        if "nobatch" in pf:
                            class_list = results.as_numpy(result_name)
                        else:
                            class_list = results.as_numpy(result_name)[b]

                        tester.assertEqual(len(class_list), num_classes)
                        if batch_size == 1:
                            expected0_flatten = output0_array.flatten()
                            expected1_flatten = output1_array.flatten()
                        else:
                            expected0_flatten = output0_array[b].flatten()
                            expected1_flatten = output1_array[b].flatten()

                        for idx, class_label in enumerate(class_list):

                            if type(class_label) == str:
                                ctuple = class_label.split(":")
                            else:
                                ctuple = "".join(chr(x) for x in class_label).split(":")
                            cval = float(ctuple[0])
                            cidx = int(ctuple[1])
                            if result_name == OUTPUT0:
                                tester.assertEqual(cval, expected0_flatten[cidx])
                                tester.assertEqual(
                                    cval, expected0_flatten[expected0_sort_idx[b][idx]]
                                )
                                if cidx == expected0_sort_idx[b][idx]:
                                    tester.assertEqual(
                                        ctuple[2].strip("\r"),
                                        "label{}".format(expected0_sort_idx[b][idx]),
                                    )
                            elif result_name == OUTPUT1:
                                tester.assertEqual(cval, expected1_flatten[cidx])
                                tester.assertEqual(
                                    cval, expected1_flatten[expected1_sort_idx[b][idx]]
                                )
                            else:
                                tester.assertTrue(
                                    False,
                                    "unexpected class result {}".format(result_name),
                                )
    finally:

        su.unregister_cleanup_shm_regions(
            shm_regions,
            shm_handles,
            precreated_shm_regions,
            outputs,
            use_system_shared_memory,
            use_cuda_shared_memory,
        )

    return results


def infer_shape_tensor(
    tester,
    pf,
    tensor_dtype,
    input_shape_values,
    dummy_input_shapes,
    use_http=True,
    use_grpc=True,
    use_streaming=True,
    shm_suffix="",
    use_system_shared_memory=False,
    priority=0,
    timeout_us=0,
    batch_size=1,
    shape_tensor_input_dtype=np.int32,
):

    if use_system_shared_memory:
        import tritonclient.utils.shared_memory as shm

    tester.assertTrue(use_http or use_grpc or use_streaming)
    tester.assertTrue(pf.startswith("plan"))
    tester.assertEqual(len(input_shape_values), len(dummy_input_shapes))

    configs = []
    if use_http:
        configs.append((f"{_tritonserver_ipaddr}:8000", "http", False))
    if use_grpc:
        configs.append((f"{_tritonserver_ipaddr}:8001", "grpc", False))
    if use_streaming:
        configs.append((f"{_tritonserver_ipaddr}:8001", "grpc", True))

    io_cnt = len(input_shape_values)

    input_shm_handle_list = []
    output_shm_handle_list = []
    dummy_input_list = []
    input_list = []
    expected_dict = dict()

    for io_num in range(io_cnt):
        dummy_input_name = "DUMMY_INPUT{}".format(io_num)
        input_name = "INPUT{}".format(io_num)
        dummy_output_name = "DUMMY_OUTPUT{}".format(io_num)
        output_name = "OUTPUT{}".format(io_num)

        rtensor_dtype = _range_repr_dtype(tensor_dtype)
        if rtensor_dtype != bool:
            dummy_in0 = np.random.randint(
                low=np.iinfo(rtensor_dtype).min,
                high=np.iinfo(rtensor_dtype).max,
                size=dummy_input_shapes[io_num],
                dtype=rtensor_dtype,
            )
        else:
            dummy_in0 = np.random.choice(
                a=[False, True], size=dummy_input_shapes[io_num]
            )
        if tensor_dtype != np.object_:
            dummy_in0 = dummy_in0.astype(tensor_dtype)
        else:
            dummy_in0 = np.array(
                [str(x) for x in dummy_in0.flatten()], dtype=object
            ).reshape(dummy_in0.shape)
        dummy_input_list.append(dummy_in0)

        in0 = np.asarray(input_shape_values[io_num], dtype=shape_tensor_input_dtype)
        input_list.append(in0)

        expected_dict[output_name] = np.ndarray.copy(in0)

        input_byte_size = in0.size * np.dtype(shape_tensor_input_dtype).itemsize
        output_byte_size = input_byte_size * batch_size
        if shape_tensor_input_dtype == np.int32:

            output_byte_size = output_byte_size * 2
        if use_system_shared_memory:
            input_shm_handle_list.append(
                (
                    shm.create_shared_memory_region(
                        input_name + shm_suffix,
                        "/" + input_name + shm_suffix,
                        input_byte_size,
                    ),
                    input_byte_size,
                )
            )
            output_shm_handle_list.append(
                (
                    shm.create_shared_memory_region(
                        output_name + shm_suffix,
                        "/" + output_name + shm_suffix,
                        output_byte_size,
                    ),
                    output_byte_size,
                )
            )
            shm.set_shared_memory_region(
                input_shm_handle_list[-1][0],
                [
                    in0,
                ],
            )

    model_name = tu.get_zero_model_name(pf, io_cnt, tensor_dtype)
    model_name = model_name + "_" + np.dtype(shape_tensor_input_dtype).name

    for config in configs:
        client_utils = grpcclient if config[1] == "grpc" else httpclient
        triton_client = client_utils.InferenceServerClient(config[0], verbose=True)

        inputs = []
        outputs = []

        for io_num in range(io_cnt):
            dummy_input_name = "DUMMY_INPUT{}".format(io_num)
            input_name = "INPUT{}".format(io_num)
            dummy_output_name = "DUMMY_OUTPUT{}".format(io_num)
            output_name = "OUTPUT{}".format(io_num)

            inputs.append(
                client_utils.InferInput(
                    dummy_input_name,
                    dummy_input_shapes[io_num],
                    np_to_triton_dtype(tensor_dtype),
                )
            )
            inputs.append(
                client_utils.InferInput(
                    input_name,
                    input_list[io_num].shape,
                    np_to_triton_dtype(shape_tensor_input_dtype),
                )
            )
            outputs.append(client_utils.InferRequestedOutput(dummy_output_name))
            outputs.append(client_utils.InferRequestedOutput(output_name))

            inputs[-2].set_data_from_numpy(dummy_input_list[io_num])
            if not use_system_shared_memory:
                inputs[-1].set_data_from_numpy(input_list[io_num])
            else:
                input_byte_size = input_shm_handle_list[io_num][1]
                output_byte_size = output_shm_handle_list[io_num][1]
                triton_client.register_system_shared_memory(
                    input_name + shm_suffix,
                    "/" + input_name + shm_suffix,
                    input_byte_size,
                )
                triton_client.register_system_shared_memory(
                    output_name + shm_suffix,
                    "/" + output_name + shm_suffix,
                    output_byte_size,
                )
                inputs[-1].set_shared_memory(input_name + shm_suffix, input_byte_size)
                outputs[-1].set_shared_memory(
                    output_name + shm_suffix, output_byte_size
                )

        if config[2]:
            user_data = UserData()
            triton_client.start_stream(partial(completion_callback, user_data))
            try:
                results = triton_client.async_stream_infer(
                    model_name,
                    inputs,
                    outputs=outputs,
                    priority=priority,
                    timeout=timeout_us,
                )
            except Exception as e:
                triton_client.stop_stream()
                raise e
            triton_client.stop_stream()
            (results, error) = user_data._completed_requests.get()
            if error is not None:
                raise error
        else:
            try:
                results = triton_client.infer(
                    model_name,
                    inputs,
                    outputs=outputs,
                    priority=priority,
                    timeout=timeout_us,
                )
            except Exception as e:
                if use_system_shared_memory:
                    for io_num in range(io_cnt):
                        shm.destroy_shared_memory_region(
                            input_shm_handle_list[io_num][0]
                        )
                        triton_client.unregister_system_shared_memory(
                            f"INPUT{io_num}" + shm_suffix
                        )
                        shm.destroy_shared_memory_region(
                            output_shm_handle_list[io_num][0]
                        )
                        triton_client.unregister_system_shared_memory(
                            f"OUTPUT{io_num}" + shm_suffix
                        )
                raise e

        for io_num in range(io_cnt):
            output_name = "OUTPUT{}".format(io_num)
            dummy_output_name = "DUMMY_OUTPUT{}".format(io_num)
            expected = expected_dict[output_name]

            dummy_out = results.as_numpy(dummy_output_name)
            if not use_system_shared_memory:
                out = results.as_numpy(output_name)
            else:
                output = results.get_output(output_name)
                if config[1] == "grpc":
                    output_shape = output.shape
                else:
                    output_shape = output["shape"]

                out = shm.get_contents_as_numpy(
                    output_shm_handle_list[io_num][0], np.int64, output_shape
                )

            if len(out.shape) == 2:

                tester.assertTrue(
                    np.array_equal(dummy_out.shape[1:], out[0]),
                    "{}, {} shape, expected: {}, got {}".format(
                        model_name, dummy_output_name, out[0], dummy_out.shape[1:]
                    ),
                )
                for b in range(1, out.shape[0]):
                    tester.assertTrue(
                        np.array_equal(out[b - 1], out[b]),
                        "expect shape tensor has consistent value, "
                        "expected: {}, got {}".format(out[b - 1], out[b]),
                    )
                out = out[0]
            else:
                tester.assertTrue(
                    np.array_equal(dummy_out.shape, out),
                    "{}, {} shape, expected: {}, got {}".format(
                        model_name, dummy_output_name, out, dummy_out.shape
                    ),
                )
            tester.assertTrue(
                np.array_equal(out, expected),
                "{}, {}, expected: {}, got {}".format(
                    model_name, output_name, expected, out
                ),
            )

            if use_system_shared_memory:
                triton_client.unregister_system_shared_memory(input_name + shm_suffix)
                triton_client.unregister_system_shared_memory(output_name + shm_suffix)

    for handle in input_shm_handle_list:
        shm.destroy_shared_memory_region(handle[0])
    for handle in output_shm_handle_list:
        shm.destroy_shared_memory_region(handle[0])


def infer_zero(
    tester,
    pf,
    batch_size,
    tensor_dtype,
    input_shapes,
    output_shapes,
    model_version=None,
    use_http=True,
    use_grpc=True,
    use_http_json_tensors=True,
    use_streaming=True,
    shm_region_name_prefix=None,
    use_system_shared_memory=False,
    use_cuda_shared_memory=False,
    priority=0,
    timeout_us=0,
    override_model_name=None,
    override_input_names=[],
    override_output_names=[],
):

    if use_system_shared_memory:
        import tritonclient.utils.shared_memory as shm
    if use_cuda_shared_memory:
        import tritonclient.utils.cuda_shared_memory as cudashm

    tester.assertTrue(use_http or use_grpc or use_streaming)
    configs = []
    if use_http:
        configs.append((f"{_tritonserver_ipaddr}:8000", "http", False, True))
        if use_http_json_tensors and (tensor_dtype != np.float16):
            configs.append((f"{_tritonserver_ipaddr}:8000", "http", False, False))
    if use_grpc:
        configs.append((f"{_tritonserver_ipaddr}:8001", "grpc", False, False))
    if use_streaming:
        configs.append((f"{_tritonserver_ipaddr}:8001", "grpc", True, False))
    tester.assertEqual(len(input_shapes), len(output_shapes))
    io_cnt = len(input_shapes)

    if shm_region_name_prefix is None:
        shm_region_name_prefix = ["input", "output"]

    input_dict = {}
    expected_dict = {}
    shm_ip_handles = list()
    shm_op_handles = list()

    if override_model_name is None:
        model_name = tu.get_zero_model_name(pf, io_cnt, tensor_dtype)
    else:
        model_name = override_model_name
    if configs[0][1] == "http":
        metadata_client = httpclient.InferenceServerClient(configs[0][0], verbose=True)
        metadata = metadata_client.get_model_metadata(model_name)
        platform = metadata["platform"]
    else:
        metadata_client = grpcclient.InferenceServerClient(configs[0][0], verbose=True)
        metadata = metadata_client.get_model_metadata(model_name)
        platform = metadata.platform

    for io_num in range(io_cnt):
        if override_input_names:
            input_name = override_input_names[io_num]
        else:
            if platform == "pytorch_libtorch":
                input_name = "INPUT__{}".format(io_num)
            else:
                input_name = "INPUT{}".format(io_num)

        if override_output_names:
            output_name = override_output_names[io_num]
        else:
            if platform == "pytorch_libtorch":
                output_name = "OUTPUT__{}".format(io_num)
            else:
                output_name = "OUTPUT{}".format(io_num)

        input_shape = input_shapes[io_num]
        output_shape = output_shapes[io_num]

        rtensor_dtype = _range_repr_dtype(tensor_dtype)
        if rtensor_dtype != bool:
            input_array = np.random.randint(
                low=np.iinfo(rtensor_dtype).min,
                high=np.iinfo(rtensor_dtype).max,
                size=input_shape,
                dtype=rtensor_dtype,
            )
        else:
            input_array = np.random.choice(a=[False, True], size=input_shape)
        if tensor_dtype != np.object_:
            input_array = input_array.astype(tensor_dtype)
            expected_array = np.ndarray.copy(input_array)
        else:
            expected_array = np.array(
                [unicode(str(x), encoding="utf-8") for x in input_array.flatten()],
                dtype=object,
            )
            input_array = np.array(
                [str(x) for x in input_array.flatten()], dtype=object
            ).reshape(input_array.shape)

        expected_array = expected_array.reshape(output_shape)
        expected_dict[output_name] = expected_array

        if tensor_dtype == np.object_:
            output_byte_size = serialized_byte_size(expected_array)
        else:
            output_byte_size = expected_array.nbytes

        if batch_size == 1:
            input_list = [input_array]
        else:
            input_list = [x for x in input_array]

        if tensor_dtype == np.object_:
            input_list_tmp = serialize_byte_tensor_list(input_list)
        else:
            input_list_tmp = input_list

        if tensor_dtype == np.object_:
            input_byte_size = sum([serialized_byte_size(ip) for ip in input_list_tmp])
        else:
            input_byte_size = sum([ip.nbytes for ip in input_list_tmp])

        shm_io_handles = su.create_set_either_shm_region(
            [
                shm_region_name_prefix[0] + str(io_num),
                shm_region_name_prefix[1] + str(io_num),
            ],
            input_list_tmp,
            input_byte_size,
            output_byte_size,
            use_system_shared_memory,
            use_cuda_shared_memory,
        )

        if len(shm_io_handles) != 0:
            shm_ip_handles.append(shm_io_handles[0])
            shm_op_handles.append(shm_io_handles[1])
        input_dict[input_name] = input_array

    if model_version is not None:
        model_version = str(model_version)
    else:
        model_version = ""

    for config in configs:
        if config[1] == "http":
            triton_client = httpclient.InferenceServerClient(config[0], verbose=True)
        else:
            triton_client = grpcclient.InferenceServerClient(config[0], verbose=True)

        inputs = []
        output_req = []
        for io_num, (input_name, output_name) in enumerate(
            zip(input_dict.keys(), expected_dict.keys())
        ):
            input_data = input_dict[input_name]
            output_data = expected_dict[output_name]
            if tensor_dtype == np.object_:
                input_byte_size = serialized_byte_size(
                    serialize_byte_tensor(input_data)
                )
                output_byte_size = serialized_byte_size(
                    serialize_byte_tensor(output_data)
                )
            else:
                input_byte_size = input_data.nbytes
                output_byte_size = output_data.nbytes
            if config[1] == "http":
                inputs.append(
                    httpclient.InferInput(
                        input_name, input_data.shape, np_to_triton_dtype(tensor_dtype)
                    )
                )
                output_req.append(
                    httpclient.InferRequestedOutput(output_name, binary_data=config[3])
                )
            else:
                inputs.append(
                    grpcclient.InferInput(
                        input_name, input_data.shape, np_to_triton_dtype(tensor_dtype)
                    )
                )
                output_req.append(grpcclient.InferRequestedOutput(output_name))

            if not (use_cuda_shared_memory or use_system_shared_memory):
                if config[1] == "http":
                    inputs[-1].set_data_from_numpy(input_data, binary_data=config[3])
                else:
                    inputs[-1].set_data_from_numpy(input_data)
            else:

                su.register_add_either_shm_regions(
                    inputs,
                    output_req,
                    shm_region_name_prefix,
                    (shm_ip_handles, shm_op_handles),
                    io_num,
                    input_byte_size,
                    output_byte_size,
                    use_system_shared_memory,
                    use_cuda_shared_memory,
                    triton_client,
                )

        if config[2]:
            user_data = UserData()
            triton_client.start_stream(partial(completion_callback, user_data))
            try:
                results = triton_client.async_stream_infer(
                    model_name,
                    inputs,
                    model_version=model_version,
                    outputs=output_req,
                    request_id=str(_unique_request_id()),
                    priority=priority,
                    timeout=timeout_us,
                )
            except Exception as e:
                triton_client.stop_stream()
                raise e
            triton_client.stop_stream()
            (results, error) = user_data._completed_requests.get()
            if error is not None:
                raise error
        else:
            results = triton_client.infer(
                model_name,
                inputs,
                model_version=model_version,
                outputs=output_req,
                request_id=str(_unique_request_id()),
                priority=priority,
                timeout=timeout_us,
            )

        last_response = results.get_response()

        if config[1] == "http":
            response_model_name = last_response["model_name"]
            if model_version != "":
                response_model_version = last_response["model_version"]
            response_outputs = last_response["outputs"]
        else:
            response_model_name = last_response.model_name
            if model_version != "":
                response_model_version = last_response.model_version
            response_outputs = last_response.outputs

        tester.assertEqual(response_model_name, model_name)

        if model_version != "":
            tester.assertEqual(response_model_version, model_version)

        tester.assertEqual(len(response_outputs), io_cnt)

        for result in response_outputs:
            if config[1] == "http":
                result_name = result["name"]
            else:
                result_name = result.name

            tester.assertIn(result_name, expected_dict)
            if use_system_shared_memory or use_cuda_shared_memory:
                if platform == "pytorch_libtorch":
                    io_num = int(result_name.split("OUTPUT__")[1])
                else:
                    io_num = int(result_name.split("OUTPUT")[1])
                shm_handle = shm_op_handles[io_num]

                output = results.get_output(result_name)
                if config[1] == "http":
                    output_datatype = output["datatype"]
                    output_shape = output["shape"]
                else:
                    output_datatype = output.datatype
                    output_shape = output.shape
                output_dtype = triton_to_np_dtype(output_datatype)
            if use_system_shared_memory:
                output_data = shm.get_contents_as_numpy(
                    shm_handle, output_dtype, output_shape
                )
            elif use_cuda_shared_memory:
                output_data = cudashm.get_contents_as_numpy(
                    shm_handle, output_dtype, output_shape
                )
            else:
                output_data = results.as_numpy(result_name)

                if (output_data.dtype == np.object_) and (config[3] == False):
                    if config[1] == "http":
                        output_data = np.array(
                            [
                                unicode(str(x), encoding="utf-8")
                                for x in (output_data.flatten())
                            ],
                            dtype=np.object_,
                        ).reshape(output_data.shape)
                    elif config[1] == "grpc":
                        output_data = np.array(
                            [x for x in (output_data.flatten())], dtype=np.object_
                        ).reshape(output_data.shape)

            expected = expected_dict[result_name]
            tester.assertEqual(output_data.shape, expected.shape)
            tester.assertTrue(
                np.array_equal(output_data, expected),
                "{}, {}, expected: {}, got {}".format(
                    model_name, result_name, expected, output_data
                ),
            )

    if len(shm_ip_handles) != 0:
        for io_num in range(io_cnt):
            if use_cuda_shared_memory:
                triton_client.unregister_cuda_shared_memory(
                    shm_region_name_prefix[0] + str(io_num) + "_data"
                )
                triton_client.unregister_cuda_shared_memory(
                    shm_region_name_prefix[0] + str(io_num) + "_data"
                )
                cudashm.destroy_shared_memory_region(shm_ip_handles[io_num])
                cudashm.destroy_shared_memory_region(shm_op_handles[io_num])
            else:
                triton_client.unregister_system_shared_memory(
                    shm_region_name_prefix[1] + str(io_num) + "_data"
                )
                triton_client.unregister_system_shared_memory(
                    shm_region_name_prefix[1] + str(io_num) + "_data"
                )
                shm.destroy_shared_memory_region(shm_ip_handles[io_num])
                shm.destroy_shared_memory_region(shm_op_handles[io_num])

    return results


def shm_basic_infer(
    tester,
    triton_client,
    shm_ip0_handle,
    shm_ip1_handle,
    shm_op0_handle,
    shm_op1_handle,
    error_msg,
    big_shm_name="",
    big_shm_size=64,
    default_shm_byte_size=64,
    shm_output_offset=0,
    shm_output_byte_size=64,
    protocol="http",
    use_system_shared_memory=False,
    use_cuda_shared_memory=False,
):

    if use_system_shared_memory:
        import tritonclient.utils.shared_memory as shm
    elif use_cuda_shared_memory:
        import tritonclient.utils.cuda_shared_memory as cudashm
    else:
        raise Exception("No shared memory type specified")

    input0_data = np.arange(start=0, stop=16, dtype=np.int32)
    input1_data = np.ones(shape=16, dtype=np.int32)
    inputs = []
    outputs = []
    if protocol == "http":
        inputs.append(httpclient.InferInput("INPUT0", [1, 16], "INT32"))
        inputs.append(httpclient.InferInput("INPUT1", [1, 16], "INT32"))
        outputs.append(httpclient.InferRequestedOutput("OUTPUT0", binary_data=True))
        outputs.append(httpclient.InferRequestedOutput("OUTPUT1", binary_data=False))
    else:
        inputs.append(grpcclient.InferInput("INPUT0", [1, 16], "INT32"))
        inputs.append(grpcclient.InferInput("INPUT1", [1, 16], "INT32"))
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT0"))
        outputs.append(grpcclient.InferRequestedOutput("OUTPUT1"))

    inputs[0].set_shared_memory("input0_data", default_shm_byte_size)

    if type(shm_ip1_handle) == np.array:
        inputs[1].set_data_from_numpy(input0_data, binary_data=True)
    elif big_shm_name != "":
        inputs[1].set_shared_memory(big_shm_name, big_shm_size)
    else:
        inputs[1].set_shared_memory("input1_data", default_shm_byte_size)

    outputs[0].set_shared_memory(
        "output0_data", shm_output_byte_size, offset=shm_output_offset
    )
    outputs[1].set_shared_memory(
        "output1_data", shm_output_byte_size, offset=shm_output_offset
    )

    try:
        results = triton_client.infer(
            "simple", inputs, model_version="", outputs=outputs
        )
        output = results.get_output("OUTPUT0")
        if protocol == "http":
            output_datatype = output["datatype"]
            output_shape = output["shape"]
        else:
            output_datatype = output.datatype
            output_shape = output.shape
        output_dtype = triton_to_np_dtype(output_datatype)

        if use_system_shared_memory:
            output_data = shm.get_contents_as_numpy(
                shm_op0_handle, output_dtype, output_shape
            )
        elif use_cuda_shared_memory:
            output_data = cudashm.get_contents_as_numpy(
                shm_op0_handle, output_dtype, output_shape
            )

        tester.assertTrue(
            (output_data[0] == (input0_data + input1_data)).all(),
            "Model output does not match expected output",
        )
    except Exception as ex:
        error_msg.append(str(ex))
