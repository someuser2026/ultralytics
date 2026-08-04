# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""RHINO-only dataset formatting."""

import torch
import torch.nn.functional as F

from ultralytics.data.augment import Format
from ultralytics.models.rtdetr.val import RTDETRDataset


class RHINODataset(RTDETRDataset):
    """RT-DETR dataset with the RHINO ``le90`` target boundary."""

    _SPATIAL_BATCH_KEYS = (
        "img",
        "land_water_mask",
        "shoreline_distance_map",
        "shoreline_proximity_field",
    )

    def build_transforms(self, hyp=None):
        """Use the parent pipeline and change only its final RHINO ``Format`` instance."""
        transforms = super().build_transforms(hyp)
        for transform in reversed(transforms.transforms):
            if isinstance(transform, Format):
                transform.angle_mode = "le90"
                break
        else:
            raise RuntimeError("RHINODataset requires a final Format transform.")
        return transforms

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """Bottom/right-pad a RHINO batch and expose its valid image region."""
        if not batch:
            return {}

        samples = [sample.copy() for sample in batch]
        image_shapes = []
        for sample in samples:
            image = sample["img"]
            if not isinstance(image, torch.Tensor) or image.ndim != 3:
                raise ValueError(f"RHINODataset expects CHW image tensors, got {type(image).__name__}.")
            image_shapes.append((int(image.shape[-2]), int(image.shape[-1])))

        max_height = max(height for height, _ in image_shapes)
        max_width = max(width for _, width in image_shapes)
        padding_masks = []
        for sample, (height, width) in zip(samples, image_shapes):
            pad_height = max_height - height
            pad_width = max_width - width
            for key in RHINODataset._SPATIAL_BATCH_KEYS:
                value = sample.get(key)
                if not isinstance(value, torch.Tensor):
                    continue
                if tuple(value.shape[-2:]) != (height, width):
                    raise ValueError(
                        f"RHINO spatial tensor {key!r} has shape {tuple(value.shape[-2:])}, "
                        f"expected {(height, width)}."
                    )
                if pad_height or pad_width:
                    sample[key] = F.pad(value, (0, pad_width, 0, pad_height), value=0)

            padding_mask = torch.ones((max_height, max_width), dtype=torch.bool)
            padding_mask[:height, :width] = False
            padding_masks.append(padding_mask)

        collated = RTDETRDataset.collate_fn(samples)
        collated["padding_mask"] = torch.stack(padding_masks)
        collated["img_shapes"] = torch.tensor(image_shapes, dtype=torch.long)
        return collated
