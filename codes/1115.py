import sys

sys.path.append("../common")

import unittest

import numpy as np
import test_util as tu
import tritonclient.http as httpclient
from PIL import Image


class InferTest(tu.TestResultCollector):
    def _preprocess(self, img, dtype):

        sample_img = img.convert("RGB")
        resized_img = sample_img.resize((224, 224), Image.BILINEAR)
        resized = np.array(resized_img)

        typed = resized.astype(dtype)
        scaled = typed - np.asarray((123, 117, 104), dtype=dtype)
        ordered = np.transpose(scaled, (2, 0, 1))

        return ordered

    def test_resnet50(self):
        try:
            triton_client = httpclient.InferenceServerClient(url="localhost:8000")
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit(1)

        image_filename = "../images/vulture.jpeg"
        model_name = "resnet50_plan"
        batch_size = 32

        img = Image.open(image_filename)
        image_data = self._preprocess(img, np.int8)
        image_data = np.expand_dims(image_data, axis=0)

        batched_image_data = image_data
        for i in range(1, batch_size):
            batched_image_data = np.concatenate(
                (batched_image_data, image_data), axis=0
            )

        inputs = [
            httpclient.InferInput("input_tensor_0", [batch_size, 3, 224, 224], "INT8")
        ]
        inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

        outputs = [
            httpclient.InferRequestedOutput("topk_layer_output_index", binary_data=True)
        ]

        results = triton_client.infer(model_name, inputs, outputs=outputs)

        output_data = results.as_numpy("topk_layer_output_index")
        print(output_data)

        EXPECTED_CLASS_INDEX = 418
        for i in range(batch_size):
            self.assertEqual(output_data[i][0][0], EXPECTED_CLASS_INDEX)


if __name__ == "__main__":
    unittest.main()
