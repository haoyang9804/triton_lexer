from contextlib import contextmanager
from copy import deepcopy
import logging
from multiprocessing import Pool
import os

import numpy as np
from sklearn.cluster import DBSCAN as dbscan
from PIL import ImageDraw

from tao_triton.python.types import KittiBbox
from tao_triton.python.utils.kitti import write_kitti_annotation
import tritonclient.grpc.model_config_pb2 as mc

logger = logging.getLogger(__name__)


@contextmanager
def pool_context(*args, **kwargs):

    pool = Pool(*args, **kwargs)
    try:
        yield pool
    finally:
        pool.terminate()


def denormalize_bounding_bboxes(
    bbox_array,
    stride,
    offset,
    bbox_norm,
    num_classes,
    scale_w,
    scale_h,
    data_format,
    model_shape,
    frames,
    this_id,
):

    boxes = deepcopy(bbox_array)
    if data_format == mc.ModelInput.FORMAT_NCHW:
        _, model_height, model_width = model_shape
    else:
        model_height, model_width, _ = model_shape
    scales = np.zeros((boxes.shape[0], 4, boxes.shape[2], boxes.shape[3])).astype(
        np.float32
    )
    for i in range(boxes.shape[0]):
        frame = frames[(this_id * boxes.shape[0] + i) % len(frames)]
        scales[i, 0, :, :].fill(float(frame.width / model_width))
        scales[i, 1, :, :].fill(float(frame.height / model_height))
        scales[i, 2, :, :].fill(float(frame.width / model_width))
        scales[i, 3, :, :].fill(float(frame.height / model_height))
    scales = np.asarray(scales).astype(np.float32)
    target_shape = boxes.shape[-2:]
    gc_centers = [(np.arange(s) * stride + offset) for s in target_shape]
    gc_centers = [s / n for s, n in zip(gc_centers, bbox_norm)]
    for n in range(num_classes):
        boxes[:, 4 * n + 0, :, :] -= gc_centers[0][:, np.newaxis] * scale_w
        boxes[:, 4 * n + 1, :, :] -= gc_centers[1] * scale_h
        boxes[:, 4 * n + 2, :, :] += gc_centers[0][:, np.newaxis] * scale_w
        boxes[:, 4 * n + 3, :, :] += gc_centers[1] * scale_h
        boxes[:, 4 * n + 0, :, :] *= -bbox_norm[0]
        boxes[:, 4 * n + 1, :, :] *= -bbox_norm[1]
        boxes[:, 4 * n + 2, :, :] *= bbox_norm[0]
        boxes[:, 4 * n + 3, :, :] *= bbox_norm[1]

        boxes[:, 4 * n + 0, :, :] = (
            np.minimum(np.maximum(boxes[:, 4 * n + 0, :, :], 0), model_width)
            * scales[:, 0, :, :]
        )
        boxes[:, 4 * n + 1, :, :] = (
            np.minimum(np.maximum(boxes[:, 4 * n + 1, :, :], 0), model_height)
            * scales[:, 1, :, :]
        )
        boxes[:, 4 * n + 2, :, :] = (
            np.minimum(np.maximum(boxes[:, 4 * n + 2, :, :], 0), model_width)
            * scales[:, 2, :, :]
        )
        boxes[:, 4 * n + 3, :, :] = (
            np.minimum(np.maximum(boxes[:, 4 * n + 3, :, :], 0), model_height)
            * scales[:, 3, :, :]
        )
    return boxes


def thresholded_indices(cov_array, num_classes, classes, cov_threshold):

    valid_indices = []
    batch_size, num_classes, _, _ = cov_array.shape
    for image_idx in range(batch_size):
        indices_per_class = []
        for class_idx in range(num_classes):
            covs = cov_array[image_idx, class_idx, :, :].flatten()
            class_indices = covs > cov_threshold[classes[class_idx]]
            indices_per_class.append(class_indices)
        valid_indices.append(indices_per_class)
    return valid_indices


def render_image(frame, image_wise_bboxes, output_image_file, box_color, linewidth=3):

    image = frame.load_image()
    draw = ImageDraw.Draw(image)
    for annotations in image_wise_bboxes:
        class_name = annotations.category
        box = annotations.box
        outline_color = (
            box_color[class_name].R,
            box_color[class_name].G,
            box_color[class_name].B,
        )
        if (box[2] - box[0]) >= 0 and (box[3] - box[1]) >= 0:
            draw.rectangle(box, outline=outline_color)
            for i in range(linewidth):
                x1 = max(0, box[0] - i)
                y1 = max(0, box[1] - i)
                x2 = min(frame.width, box[2] + i)
                y2 = min(frame.height, box[3] + i)
                draw.rectangle(box, outline=outline_color)
    image.save(output_image_file)


def iou_vectorized(rects):

    l, t, r, b = rects.T

    isect_l = np.maximum(l[:, None], l[None, :])
    isect_t = np.maximum(t[:, None], t[None, :])
    isect_r = np.minimum(r[:, None], r[None, :])
    isect_b = np.minimum(b[:, None], b[None, :])

    isect_w = np.maximum(0, isect_r - isect_l)
    isect_h = np.maximum(0, isect_b - isect_t)
    area_isect = isect_w * isect_h

    areas = (r - l) * (b - t)

    denom = areas[:, None] + areas[None, :] - area_isect

    return area_isect / (denom + 0.01)
