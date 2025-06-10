import numpy as np
import sys
import json
import cv2
import torch
import torchvision
import time


import triton_python_backend_utils as pb_utils


class TritonPythonModel:

    def initialize(self, args):

        self.model_config = model_config = json.loads(args["model_config"])

        output0_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT0")

        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config["data_type"]
        )

        output1_config = pb_utils.get_output_config_by_name(model_config, "OUTPUT1")

        self.output1_dtype = pb_utils.triton_string_to_numpy(
            output1_config["data_type"]
        )

        self.confThreshold = 0.4
        self.class_id = 0

    def xywh2xyxy(self, x):

        y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def non_max_suppression(
        self, pred, conf_thres=0.4, iou_thres=0.5, classes=0, agnostic=False
    ):

        prediction = torch.from_numpy(pred.astype(np.float32))
        if prediction.dtype is torch.float16:
            prediction = prediction.float()
        nc = prediction[0].shape[1] - 5
        xc = prediction[..., 4] > conf_thres
        min_wh, max_wh = 2, 4096
        max_det = 100
        time_limit = 10.0
        multi_label = nc > 1
        output = [None] * prediction.shape[0]
        t = time.time()
        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                continue
            x[:, 5:] *= x[:, 4:5]
            box = self.xywh2xyxy(x[:, :4])
            if multi_label:
                i, j = (x[:, 5:] > conf_thres).nonzero().t()
                x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            else:
                conf, j = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            if classes:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            n = x.shape[0]
            if not n:
                continue
            c = x[:, 5:6] * (0 if agnostic else max_wh)
            boxes, scores = x[:, :4] + c, x[:, 4]
            i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:
                i = i[:max_det]
            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                break
        return output

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):

        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
                img1_shape[0] - img0_shape[0] * gain
            ) / 2
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)
        return coords

    def clip_coords(self, boxes, img_shape):

        boxes[:, 0].clamp_(0, img_shape[1])
        boxes[:, 1].clamp_(0, img_shape[0])
        boxes[:, 2].clamp_(0, img_shape[1])
        boxes[:, 3].clamp_(0, img_shape[0])

    def get_bbox(self, image, detections_bs):
        boxes = self.non_max_suppression(detections_bs)
        image_shape = image.shape
        outputs = [[-1, 0, 0, 0, 0, 0]]
        crops = []
        if len(boxes) > 0:
            for i, det in enumerate(boxes):
                if det is not None and len(det):
                    det[:, :4] = self.scale_coords(
                        (640, 640),
                        det[:, :4],
                        (image_shape[0], image_shape[1], image_shape[2]),
                    ).round()

                    for *xyxy, conf, cls in det:
                        x_min = xyxy[0] / float(image_shape[1])
                        y_min = xyxy[1] / float(image_shape[0])
                        x_max = xyxy[2] / float(image_shape[1])
                        y_max = xyxy[3] / float(image_shape[0])
                        score = conf
                        class_id = int(cls)

                        if class_id == self.class_id and score > self.confThreshold:
                            outputs.append(
                                [class_id, score, x_min, y_min, x_max, y_max]
                            )
                            crops.append([image[y_min:y_max, x_min:x_max]])
        return outputs, crops

    def resize_image(self, input_image, target_size=416, mode=None):

        img = input_image.copy()
        (rows, cols, _) = img.shape
        if mode:
            img = cv2.resize(img, (int(target_size), int(target_size)), mode)
        else:
            img = cv2.resize(img, (int(target_size), int(target_size)))

        scale = [float(target_size) / cols, float(target_size) / rows]

        return img, scale

    def process_classfi_data(self, crops):

        new_crops = []
        for image_np in crops:
            image_np = self.resize_image(image_np, 260, "inter_area")
            image_np = image_np.astype(np.float32)
            image_np /= 255.0
            image_np -= 0.5
            image_np *= 2
            new_crops.append(image_np)

        new_crops.append(np.random.rand(260, 260, 3))
        return np.asarray(new_crops)

    def execute(self, requests):

        output0_dtype = self.output0_dtype
        output1_dtype = self.output1_dtype
        responses = []

        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0").as_numpy()
            in_0 = np.transpose(in_0, (1, 2, 0))
            in_1 = pb_utils.get_input_tensor_by_name(request, "INPUT1").as_numpy()

            bboxs, crops = self.get_bbox(in_0, in_1)
            crops = self.process_classfi_data(crops)
            out_tensor_0 = pb_utils.Tensor("OUTPUT0", crops.astype(output0_dtype))
            out_tensor_1 = pb_utils.Tensor(
                "OUTPUT1", np.asarray(bboxs).astype(output1_dtype)
            )
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[out_tensor_0, out_tensor_1]
            )
            responses.append(inference_response)
        return responses


def finalize(self):

    print("Cleaning up...")
