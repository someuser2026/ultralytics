# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""NMS-free RHINO reference postprocessing."""

from __future__ import annotations

import math

import torch

from ultralytics.utils import ops


def rhino_postprocess(
    predictions: torch.Tensor,
    image_shapes: list[tuple[int, int]],
    conf: float = 0.0,
    classes: list[int] | None = None,
    max_candidates: int = 500,
) -> list[dict[str, torch.Tensor]]:
    """Flatten query/class scores and select reference global top-k detections."""
    if predictions.ndim != 3 or predictions.shape[-1] <= 5:
        raise ValueError(f"RHINO predictions must have shape [B, Q, 5+C], got {tuple(predictions.shape)}.")
    if len(image_shapes) != predictions.shape[0]:
        raise ValueError("RHINO image_shapes must contain one (height, width) pair per batch item.")

    boxes, scores = predictions[..., :5], predictions[..., 5:]
    num_classes = scores.shape[-1]
    class_filter = (
        None
        if classes is None
        else torch.as_tensor(classes, dtype=torch.long, device=predictions.device)
    )
    outputs = []
    for image_index, image_shape in enumerate(image_shapes):
        flattened_scores = scores[image_index].flatten()
        candidate_count = min(int(max_candidates), flattened_scores.numel())
        candidate_scores, candidate_indices = flattened_scores.topk(candidate_count)
        query_indices = torch.div(candidate_indices, num_classes, rounding_mode="floor")
        labels = candidate_indices.remainder(num_classes)

        keep = candidate_scores > float(conf)
        if class_filter is not None:
            keep &= (labels[:, None] == class_filter[None]).any(1)
        candidate_scores = candidate_scores[keep]
        query_indices = query_indices[keep]
        labels = labels[keep]

        selected_boxes = boxes[image_index, query_indices].clone()
        selected_boxes[..., 4] *= math.pi
        height, width = int(image_shape[0]), int(image_shape[1])
        selected_boxes[..., [0, 2]] *= width
        selected_boxes[..., [1, 3]] *= height
        selected_boxes = ops.regularize_rboxes(selected_boxes, angle_mode="le90")
        outputs.append({"bboxes": selected_boxes, "conf": candidate_scores, "cls": labels})
    return outputs
