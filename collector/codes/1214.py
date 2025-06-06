import io
import json
import math

import cv2
import numpy as np


import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):

        model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(
            model_config, "detection_postprocessing_output"
        )

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

    def execute(self, requests):

        output0_dtype = self.output0_dtype

        responses = []

        def fourPointsTransform(frame, vertices):
            vertices = np.asarray(vertices)
            outputSize = (100, 32)
            targetVertices = np.array(
                [
                    [0, outputSize[1] - 1],
                    [0, 0],
                    [outputSize[0] - 1, 0],
                    [outputSize[0] - 1, outputSize[1] - 1],
                ],
                dtype="float32",
            )

            rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
            result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
            return result

        def decodeBoundingBoxes(scores, geometry, scoreThresh=0.5):
            detections = []
            confidences = []

            assert len(scores.shape) == 4, "Incorrect dimensions of scores"
            assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
            assert scores.shape[0] == 1, "Invalid dimensions of scores"
            assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
            assert scores.shape[1] == 1, "Invalid dimensions of scores"
            assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
            assert (
                scores.shape[2] == geometry.shape[2]
            ), "Invalid dimensions of scores and geometry"
            assert (
                scores.shape[3] == geometry.shape[3]
            ), "Invalid dimensions of scores and geometry"
            height = scores.shape[2]
            width = scores.shape[3]
            for y in range(0, height):

                scoresData = scores[0][0][y]
                x0_data = geometry[0][0][y]
                x1_data = geometry[0][1][y]
                x2_data = geometry[0][2][y]
                x3_data = geometry[0][3][y]
                anglesData = geometry[0][4][y]
                for x in range(0, width):
                    score = scoresData[x]

                    if score < scoreThresh:
                        continue

                    offsetX = x * 4.0
                    offsetY = y * 4.0
                    angle = anglesData[x]

                    cosA = math.cos(angle)
                    sinA = math.sin(angle)
                    h = x0_data[x] + x2_data[x]
                    w = x1_data[x] + x3_data[x]

                    offset = [
                        offsetX + cosA * x1_data[x] + sinA * x2_data[x],
                        offsetY - sinA * x1_data[x] + cosA * x2_data[x],
                    ]

                    p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
                    p3 = (-cosA * w + offset[0], sinA * w + offset[1])
                    center = (0.5 * (p1[0] + p3[0]), 0.5 * (p1[1] + p3[1]))
                    detections.append((center, (w, h), -1 * angle * 180.0 / math.pi))
                    confidences.append(float(score))

            return [detections, confidences]

        for request in requests:

            in_1 = pb_utils.get_input_tensor_by_name(
                request, "detection_postprocessing_input_1"
            )
            in_2 = pb_utils.get_input_tensor_by_name(
                request, "detection_postprocessing_input_2"
            )
            in_3 = pb_utils.get_input_tensor_by_name(
                request, "detection_postprocessing_input_3"
            )

            scores = in_1.as_numpy().transpose(0, 3, 1, 2)
            geometry = in_2.as_numpy().transpose(0, 3, 1, 2)
            frame = np.squeeze(in_3.as_numpy(), axis=0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            [boxes, confidences] = decodeBoundingBoxes(scores, geometry)
            indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, 0.5, 0.4)

            cropped_list = []
            cv2.imwrite("frame.png", frame)
            count = 0
            for i in indices:

                count += 1
                vertices = cv2.boxPoints(boxes[i])
                cropped = fourPointsTransform(frame, vertices)
                cv2.imwrite(str(count) + ".png", cropped)
                cropped = np.expand_dims(
                    cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), axis=0
                )

                cropped_list.append(((cropped / 255.0) - 0.5) * 2)
            cropped_arr = np.stack(cropped_list, axis=0)

            np.save("tensor.pkl", cropped_arr)
            out_tensor_0 = pb_utils.Tensor(
                "detection_postprocessing_output", cropped_arr.astype(output0_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):

        print("Cleaning up...")
