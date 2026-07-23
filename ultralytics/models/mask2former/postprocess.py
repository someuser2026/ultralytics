"""Reference Mask2Former instance selection and mask finalization."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def unpack_mask2former_outputs(preds) -> dict[str, Tensor]:
    """Normalize eager dictionaries and tensor-only exported outputs to native fields."""
    if isinstance(preds, dict):
        return preds
    if isinstance(preds, (list, tuple)) and len(preds) == 2:
        logits = next((x for x in preds if isinstance(x, Tensor) and x.ndim == 3), None)
        masks = next((x for x in preds if isinstance(x, Tensor) and x.ndim == 4), None)
        if logits is not None and masks is not None:
            return {"pred_logits": logits, "pred_masks": masks}
    raise TypeError(f"Expected native Mask2Former dict or (pred_logits, pred_masks), got {type(preds).__name__}.")


def select_mask2former_instances(
    pred_logits: Tensor,
    pred_masks: Tensor,
    num_classes: int,
    max_per_image: int,
) -> list[dict[str, Tensor]]:
    """Select the reference top query/class pairs without confidence filtering or NMS."""
    selections = []
    for logits, masks in zip(pred_logits, pred_masks):
        scores = logits.softmax(dim=-1)[:, :num_classes]
        flat_scores = scores.flatten()
        k = min(int(max_per_image), flat_scores.numel())
        class_scores, top_indices = flat_scores.topk(k, sorted=False)
        class_indices = top_indices.remainder(num_classes)
        query_indices = torch.div(top_indices, num_classes, rounding_mode="floor")
        selections.append(
            {
                "class_scores": class_scores,
                "cls": class_indices,
                "query_indices": query_indices,
                "mask_logits": masks.index_select(0, query_indices),
            }
        )
    return selections


def resize_mask_logits(mask_logits: Tensor, output_shape: tuple[int, int]) -> Tensor:
    """Bilinearly resize selected mask logits before thresholding."""
    if mask_logits.shape[-2:] == tuple(output_shape):
        return mask_logits
    return F.interpolate(mask_logits[:, None], size=output_shape, mode="bilinear", align_corners=False)[:, 0]


def mask_logits_to_boxes(binary_masks: Tensor) -> Tensor:
    """Compute reference mask2bbox boxes, returning zeros for empty masks."""
    boxes = torch.zeros((binary_masks.shape[0], 4), device=binary_masks.device, dtype=torch.float32)
    x_any = binary_masks.any(dim=1)
    y_any = binary_masks.any(dim=2)
    for i in range(binary_masks.shape[0]):
        x = torch.where(x_any[i])[0]
        y = torch.where(y_any[i])[0]
        if x.numel() and y.numel():
            boxes[i] = boxes.new_tensor((x[0], y[0], x[-1] + 1, y[-1] + 1))
    return boxes


def finalize_mask2former_instances(
    selection: dict[str, Tensor],
    mask_logits: Tensor,
    mask_threshold: float = 0.5,
) -> dict[str, Tensor]:
    """Apply reference binary masks, mask-quality rescoring, and mask-derived boxes."""
    if not 0.0 < mask_threshold < 1.0:
        raise ValueError(f"mask_threshold must be strictly between 0 and 1, received {mask_threshold}.")
    threshold_logit = math.log(mask_threshold / (1.0 - mask_threshold))
    binary_masks = mask_logits > threshold_logit
    foreground = binary_masks.to(mask_logits.dtype)
    mask_scores = (mask_logits.sigmoid() * foreground).flatten(1).sum(1) / (
        foreground.flatten(1).sum(1) + 1e-6
    )
    return {
        "bboxes": mask_logits_to_boxes(binary_masks),
        "conf": selection["class_scores"] * mask_scores,
        "cls": selection["cls"].to(mask_logits.dtype),
        "masks": binary_masks,
    }


def refine_selected_masks(
    head,
    selection: dict[str, Tensor],
    point_features: list[Tensor] | None,
    image_index: int,
    image_shape: tuple[int, int],
) -> Tensor:
    """Upsample selected masks, optionally applying the configured PointRend refiner first."""
    coarse_logits = resize_mask_logits(selection["mask_logits"], image_shape)
    if not (
        point_features is not None
        and head is not None
        and getattr(head, "point_rend_enabled", False)
        and hasattr(head, "point_rend")
        and coarse_logits.shape[0]
    ):
        return coarse_logits

    from ultralytics.nn.modules.pointrend import get_pointrend_adapter

    coarse_masks = coarse_logits > 0
    coarse_boxes = mask_logits_to_boxes(coarse_masks).to(coarse_logits)
    adapter = get_pointrend_adapter(head)
    fine = [feature[image_index : image_index + 1] for feature in point_features]
    batch_indices = torch.zeros(coarse_logits.shape[0], device=coarse_logits.device, dtype=torch.long)
    instances = adapter.from_full_logits(
        selection["mask_logits"][:, None],
        coarse_boxes,
        batch_indices,
        fine,
        image_shape,
    )
    return adapter.refined_image_logits(instances)
