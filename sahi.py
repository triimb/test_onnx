# File: rfdetr/detr.py

import numpy as np
import torch
from PIL import Image
from typing import Tuple, List
import supervision as sv
from tifffile import imread

# ------------------------------
# SAHI SLICE INFERENCE FUNCTION
# ------------------------------
def sahi_infer_on_image(
    image_path: str,
    model,
    device: str = "cuda",
    slice_size: int = 1024,
    overlap: float = 0.1,
    threshold: float = 0.5
) -> sv.Detections:
    """
    Perform SAHI-style sliced inference on a TIFF image with a custom RF-DETR model.
    Returns merged detections as a sv.Detections object.
    """
    img = load_tiff_image(image_path)
    print_slice_info(img.shape[:2], slice_size, overlap)
    slices = slice_image(img, slice_size=slice_size, overlap=overlap)

    batch = []
    positions = []
    for offset, crop in slices:
        pil_crop = Image.fromarray((crop * 255).astype(np.uint8))
        batch.append(pil_crop)
        positions.append(offset)

    detections = model.predict(batch, threshold=threshold)
    if isinstance(detections, sv.Detections):
        detections = [detections]

    mapped_detections = [map_detections(d, pos) for d, pos in zip(detections, positions)]
    final_detections = greedy_nms(mapped_detections)
    return final_detections


def load_tiff_image(path: str) -> np.ndarray:
    img = imread(path)
    if img.dtype != np.uint8:
        img = (img / img.max()).astype(np.float32)
    else:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def slice_image(img: np.ndarray, slice_size: int, overlap: float) -> List[Tuple[Tuple[int, int], np.ndarray]]:
    h, w = img.shape[:2]
    step = int(slice_size * (1 - overlap))
    slices = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            y2 = min(y + slice_size, h)
            x2 = min(x + slice_size, w)
            crop = img[y:y2, x:x2]
            slices.append(((x, y), crop))
    return slices


def map_detections(dets: sv.Detections, offset: Tuple[int, int]) -> sv.Detections:
    x_offset, y_offset = offset
    new_boxes = dets.xyxy.copy()
    new_boxes[:, [0, 2]] += x_offset
    new_boxes[:, [1, 3]] += y_offset
    return sv.Detections(
        xyxy=new_boxes,
        confidence=dets.confidence.copy(),
        class_id=dets.class_id.copy(),
    )


def greedy_nms(detections: List[sv.Detections], iou_thresh: float = 0.5) -> sv.Detections:
    all_detections = sv.Detections.merge(detections)
    return all_detections.with_nms(threshold=iou_thresh)


def print_slice_info(img_shape: Tuple[int, int], slice_size: int, overlap: float):
    h, w = img_shape
    step = int(slice_size * (1 - overlap))
    n_h = (h - 1) // step + 1
    n_w = (w - 1) // step + 1
    total = n_h * n_w
    print(f"[SAHI INFO] Image size: {w}x{h}")
    print(f"[SAHI INFO] Slice size: {slice_size}, Overlap: {overlap} --> Step: {step}")
    print(f"[SAHI INFO] Slices along W: {n_w}, along H: {n_h} → total: {total} slices")
