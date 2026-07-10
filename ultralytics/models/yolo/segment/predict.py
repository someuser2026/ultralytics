# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, ops


def _segment_head(model):
    """Return the native segmentation head through optional backend/model wrappers."""

    module = model
    for _ in range(3):
        candidate = getattr(module, "model", None)
        if candidate is None:
            break
        module = candidate
    return module[-1] if hasattr(module, "__getitem__") else None


class SegmentationPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO segmentation model.
        batch (list): Current batch of images being processed.

    Methods:
        postprocess: Apply non-max suppression and process segmentation detections.
        construct_results: Construct a list of result objects from predictions.
        construct_result: Construct a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo11n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the SegmentationPredictor with configuration, overrides, and callbacks.

        This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
        prediction results.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """
        Apply non-max suppression and process segmentation detections for each image in the input batch.

        Args:
            preds (tuple): Model predictions, containing bounding boxes, scores, classes, and mask coefficients.
            img (torch.Tensor): Input image tensor in model format, with shape (B, C, H, W).
            orig_imgs (list | torch.Tensor | np.ndarray): Original image or batch of images.

        Returns:
            (list): List of Results objects containing the segmentation predictions for each image in the batch.
                Each Results object includes both bounding boxes and segmentation masks.

        Examples:
            >>> predictor = SegmentationPredictor(overrides=dict(model="yolo11n-seg.pt"))
            >>> results = predictor.postprocess(preds, img, orig_img)
        """
        raw = None
        if isinstance(preds, (tuple, list)) and isinstance(preds[0], (tuple, list)):
            pred, protos = preds[0]
            raw = preds[1] if len(preds) > 1 and isinstance(preds[1], dict) else None
        elif isinstance(preds, (tuple, list)):
            pred, protos = preds
        else:
            raise TypeError(f"SegmentationPredictor expects tuple/list predictions, got {type(preds).__name__}.")
        point_features = raw.get("pointrend_features") if raw is not None else None
        return super().postprocess(pred, img, orig_imgs, protos=protos, pointrend_features=point_features)

    def construct_results(self, preds, img, orig_imgs, protos, pointrend_features=None):
        """
        Construct a list of result objects from the predictions.

        Args:
            preds (list[torch.Tensor]): List of predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_imgs (list[np.ndarray]): List of original images before preprocessing.
            protos (list[torch.Tensor]): List of prototype masks.

        Returns:
            (list[Results]): List of result objects containing the original images, image paths, class names,
                bounding boxes, and masks.
        """
        if pointrend_features is None:
            return [
                self.construct_result(pred, img, orig_img, img_path, proto)
                for pred, orig_img, img_path, proto in zip(preds, orig_imgs, self.batch[0], protos)
            ]
        return [
            self.construct_result(
                pred,
                img,
                orig_img,
                img_path,
                proto,
                [feature[i : i + 1] for feature in pointrend_features],
            )
            for i, (pred, orig_img, img_path, proto) in enumerate(zip(preds, orig_imgs, self.batch[0], protos))
        ]

    def construct_result(self, pred, img, orig_img, img_path, proto, pointrend_features=None):
        """
        Construct a single result object from the prediction.

        Args:
            pred (torch.Tensor): The predicted bounding boxes, scores, and masks.
            img (torch.Tensor): The image after preprocessing.
            orig_img (np.ndarray): The original image before preprocessing.
            img_path (str): The path to the original image.
            proto (torch.Tensor): The prototype masks.

        Returns:
            (Results): Result object containing the original image, image path, class names, bounding boxes, and masks.
        """
        head = _segment_head(self.model)
        use_pointrend = bool(
            pred.shape[0]
            and pointrend_features is not None
            and head is not None
            and getattr(head, "point_rend_enabled", False)
            and hasattr(head, "point_rend")
        )
        if pred.shape[0] == 0:  # save empty boxes
            masks = None
        elif use_pointrend:
            from ultralytics.nn.modules.pointrend import get_pointrend_adapter

            boxes_input = pred[:, :4].clone()
            batch_indices = torch.zeros(pred.shape[0], device=pred.device, dtype=torch.long)
            adapter = get_pointrend_adapter(head)
            if type(head).__name__ == "Mask2FormerHead":
                query_indices = pred[:, 6:].argmax(1)
                full_logits = proto.index_select(0, query_indices)[:, None]
                instances = adapter.from_full_logits(
                    full_logits,
                    boxes_input,
                    batch_indices,
                    pointrend_features,
                    tuple(img.shape[2:]),
                )
            else:
                instances = adapter.from_coefficients(
                    pred[:, 6:],
                    proto.unsqueeze(0),
                    boxes_input,
                    batch_indices,
                    pointrend_features,
                    tuple(img.shape[2:]),
                )
            masks = adapter.refined_image_logits(instances)
            if self.args.retina_masks:
                masks = ops.scale_masks(masks[None], orig_img.shape[:2])[0]
            masks = masks > 0
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        elif self.args.retina_masks:
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            masks = ops.process_mask_native(proto, pred[:, 6:], pred[:, :4], orig_img.shape[:2])  # HWC
        else:
            masks = ops.process_mask(proto, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True)  # HWC
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
        if masks is not None:
            keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
            pred, masks = pred[keep], masks[keep]
        return Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
