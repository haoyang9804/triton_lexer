import numpy as np
import json
import triton_python_backend_utils as pb_utils
import cv2


class TritonPythonModel:

    def initialize(self, args):

        self.model_config = model_config = json.loads(args["model_config"])

        num_detections_config = pb_utils.get_output_config_by_name(
            model_config, "num_detections"
        )
        detection_boxes_config = pb_utils.get_output_config_by_name(
            model_config, "detection_boxes"
        )

        detection_scores_config = pb_utils.get_output_config_by_name(
            model_config, "detection_scores"
        )

        detection_classes_config = pb_utils.get_output_config_by_name(
            model_config, "detection_classes"
        )

        self.num_detections_dtype = pb_utils.triton_string_to_numpy(
            num_detections_config["data_type"]
        )

        self.detection_boxes_dtype = pb_utils.triton_string_to_numpy(
            detection_boxes_config["data_type"]
        )

        self.detection_scores_dtype = pb_utils.triton_string_to_numpy(
            detection_scores_config["data_type"]
        )

        self.detection_classes_dtype = pb_utils.triton_string_to_numpy(
            detection_classes_config["data_type"]
        )

        self.score_threshold = 0.25
        self.nms_threshold = 0.45

    def execute(self, requests):

        num_detections_dtype = self.num_detections_dtype
        detection_boxes_dtype = self.detection_boxes_dtype
        detection_scores_dtype = self.detection_scores_dtype
        detection_classes_dtype = self.detection_classes_dtype

        responses = []

        for request in requests:

            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_0")

            outputs = in_0.as_numpy()

            outputs = np.array([cv2.transpose(outputs[0])])
            rows = outputs.shape[1]

            boxes = []
            scores = []
            class_ids = []
            for i in range(rows):
                classes_scores = outputs[0][i][4:]
                (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(
                    classes_scores
                )
                if maxScore >= self.score_threshold:
                    box = [
                        outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                        outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                        outputs[0][i][2],
                        outputs[0][i][3],
                    ]
                    boxes.append(box)
                    scores.append(maxScore)
                    class_ids.append(maxClassIndex)

            result_boxes = cv2.dnn.NMSBoxes(
                boxes, scores, self.score_threshold, self.nms_threshold, 0.5
            )

            num_detections = 0
            output_boxes = []
            output_scores = []
            output_classids = []
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                detection = {
                    "class_id": class_ids[index],
                    "confidence": scores[index],
                    "box": box,
                }
                output_boxes.append(box)
                output_scores.append(scores[index])
                output_classids.append(class_ids[index])

                num_detections += 1

            num_detections = np.array(num_detections)
            num_detections = pb_utils.Tensor(
                "num_detections", num_detections.astype(num_detections_dtype)
            )

            detection_boxes = np.array(output_boxes)
            detection_boxes = pb_utils.Tensor(
                "detection_boxes", detection_boxes.astype(detection_boxes_dtype)
            )

            detection_scores = np.array(output_scores)
            detection_scores = pb_utils.Tensor(
                "detection_scores", detection_scores.astype(detection_scores_dtype)
            )
            detection_classes = np.array(output_classids)
            detection_classes = pb_utils.Tensor(
                "detection_classes", detection_classes.astype(detection_classes_dtype)
            )

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    num_detections,
                    detection_boxes,
                    detection_scores,
                    detection_classes,
                ]
            )
            responses.append(inference_response)

        return responses

    def finalize(self):

        pass
