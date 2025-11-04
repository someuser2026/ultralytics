# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import ops, DEFAULT_CFG


class RTDETRPredictor(BasePredictor):
    """
    RT-DETR (Real-Time Detection Transformer) Predictor extending the BasePredictor class for making predictions.

    This class leverages Vision Transformers to provide real-time object detection while maintaining high accuracy.
    It supports key features like efficient hybrid encoding and IoU-aware query selection.

    Attributes:
        imgsz (int): Image size for inference (must be square and scale-filled).
        args (dict): Argument overrides for the predictor.
        model (torch.nn.Module): The loaded RT-DETR model.
        batch (list): Current batch of processed inputs.

    Methods:
        postprocess: Postprocess raw model predictions to generate bounding boxes and confidence scores.
        pre_transform: Pre-transform input images before feeding them into the model for inference.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.rtdetr import RTDETRPredictor
        >>> args = dict(model="rtdetr-l.pt", source=ASSETS)
        >>> predictor = RTDETRPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def postprocess(self, preds, img, orig_imgs):
        """
        Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

        The method filters detections based on confidence and class if specified in `self.args`. It converts
        model predictions to Results objects containing properly scaled bounding boxes.

        Args:
            preds (list | tuple): List of [predictions, extra] from the model, where predictions contain
                bounding boxes and scores.
            img (torch.Tensor): Processed input images with shape (N, 3, H, W).
            orig_imgs (list | torch.Tensor): Original, unprocessed images.

        Returns:
            results (list[Results]): A list of Results objects containing the post-processed bounding boxes,
                confidence scores, and class labels.
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        nd = preds[0].shape[-1]
        bboxes, scores = preds[0].split((4, nd - 4), dim=-1)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for bbox, score, orig_img, img_path in zip(bboxes, scores, orig_imgs, self.batch[0]):  # (300, 4)
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
            idx = max_score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]  # filter
            pred = pred[pred[:, 4].argsort(descending=True)][: self.args.max_det]
            oh, ow = orig_img.shape[:2]
            pred[..., [0, 2]] *= ow  # scale x coordinates to original width
            pred[..., [1, 3]] *= oh  # scale y coordinates to original height
            results.append(Results(orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def pre_transform(self, im):
        """
        Pre-transform input images before feeding them into the model for inference.

        The input images are letterboxed to ensure a square aspect ratio and scale-filled. The size must be square
        (640) and scale_filled.

        Args:
            im (list[np.ndarray]  | torch.Tensor): Input images of shape (N, 3, H, W) for tensor,
                [(H, W, 3) x N] for list.

        Returns:
            (list): List of pre-transformed images ready for model inference.
        """
        letterbox = LetterBox(self.imgsz, auto=False, scale_fill=True)
        return [letterbox(image=x) for x in im]

class RTDETRSegmentPredictor(RTDETRPredictor):
    """
    RT-DETR Segment Predictor extending the RTDETRPredictor class for making predictions.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Argument overrides for the predictor.
        model (torch.nn.Module): The loaded RT-DETR model.
        batch (list): Current batch of processed inputs.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the RTDETRSegmentPredictor with configuration, overrides, and callbacks.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "segment"

    def postprocess(self, preds, img, orig_imgs):
        """
        Postprocess the raw predictions from the model to generate bounding boxes, confidence scores, and masks.

        Args:
            preds (list | tuple): List of [predictions, extra] from the model, where predictions contain
                bounding boxes, scores, and mask coefficients. Extra contains protos.
            img (torch.Tensor): Processed input images with shape (N, 3, H, W).
            orig_imgs (list | torch.Tensor): Original, unprocessed images.

        Returns:
            results (list[Results]): A list of Results objects containing the post-processed bounding boxes,
                confidence scores, class labels, and masks.
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        # Extract protos from extra output
        if isinstance(preds[1], (list, tuple)) and len(preds[1]) > 6:
            protos = preds[1][6]  # Protos is at index 6 in the tuple
        elif preds[1] is not None:
            protos = preds[1] if isinstance(preds[1], torch.Tensor) else None
        else:
            protos = None

        nd = preds[0].shape[-1]
        # Split: bboxes (4), scores (nc), masks (nm)
        nm = getattr(self.model.model[-1], "nm", 32) if hasattr(self.model, "model") else 32
        nc = len(self.model.names) if hasattr(self.model, "names") else nd - 4 - nm
        bboxes = preds[0][..., :4]
        scores = preds[0][..., 4 : 4 + nc]
        mask_coeffs = preds[0][..., 4 + nc :]

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for bbox, score, mask_coeff, orig_img, img_path in zip(
            bboxes, scores, mask_coeffs, orig_imgs, self.batch[0]
        ):
            bbox = ops.xywh2xyxy(bbox)
            max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
            idx = max_score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx
            pred = torch.cat([bbox, max_score, cls, mask_coeff], dim=-1)[idx]  # filter
            pred = pred[pred[:, 4].argsort(descending=True)][: self.args.max_det]
            oh, ow = orig_img.shape[:2]
            pred[:, [0, 2]] *= ow  # scale x coordinates to original width
            pred[:, [1, 3]] *= oh  # scale y coordinates to original height

            # Process masks
            masks = None
            if protos is not None and pred.shape[0] > 0:
                if self.args.retina_masks:
                    masks = ops.process_mask_native(
                        protos, pred[:, 6:], pred[:, :4], orig_img.shape[:2]
                    )  # HWC
                else:
                    masks = ops.process_mask(
                        protos, pred[:, 6:], pred[:, :4], img.shape[2:], upsample=True
                    )  # HWC
                    pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)

            if masks is not None:
                keep = masks.sum((-2, -1)) > 0  # only keep predictions with masks
                pred, masks = pred[keep], masks[keep]

            results.append(
                Results(orig_img, path=img_path, names=self.model.names, boxes=pred[:, :6], masks=masks)
            )
        return results


class RTDETROBBPredictor(RTDETRPredictor):
    """
    RT-DETR OBB Predictor extending the RTDETRPredictor class for making predictions with oriented bounding boxes.

    This class specializes in processing OBB model outputs, handling rotated bounding boxes with angles.

    Attributes:
        args (dict): Argument overrides for the predictor.
        model (torch.nn.Module): The loaded RT-DETR model.
        batch (list): Current batch of processed inputs.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the RTDETROBBPredictor with configuration, overrides, and callbacks.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "obb"

    def postprocess(self, preds, img, orig_imgs):
        """
        Postprocess the raw predictions from the model to generate oriented bounding boxes and confidence scores.

        Args:
            preds (list | tuple): List of [predictions, extra] from the model, where predictions contain
                rotated bounding boxes (5D: x, y, w, h, angle) and scores.
            img (torch.Tensor): Processed input images with shape (N, 3, H, W).
            orig_imgs (list | torch.Tensor): Original, unprocessed images.

        Returns:
            results (list[Results]): A list of Results objects containing the post-processed oriented bounding boxes,
                confidence scores, and class labels.
        """
        if not isinstance(preds, (list, tuple)):  # list for PyTorch inference but list[0] Tensor for export inference
            preds = [preds, None]

        nd = preds[0].shape[-1]
        rboxes, scores = preds[0].split((5, nd - 5), dim=-1)

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for rbox, score, orig_img, img_path in zip(rboxes, scores, orig_imgs, self.batch[0]):  # (300, 5)
            max_score, cls = score.max(-1, keepdim=True)  # (300, 1)
            idx = max_score.squeeze(-1) > self.args.conf  # (300, )
            if self.args.classes is not None:
                idx = (cls == torch.tensor(self.args.classes, device=cls.device)).any(1) & idx

            # Regularize and scale rboxes
            rboxes_reg = ops.regularize_rboxes(torch.cat([rbox[:, :4], rbox[:, -1:]], dim=-1))
            oh, ow = orig_img.shape[:2]
            rboxes_reg[:, :4] = ops.scale_boxes(
                img.shape[2:], rboxes_reg[:, :4], orig_img.shape, xywh=True
            )

            # Combine rbox, score, and class
            pred = torch.cat([rboxes_reg, max_score, cls], dim=-1)[idx]  # filter
            pred = pred[pred[:, 5].argsort(descending=True)][: self.args.max_det]
            obb = pred[:, :6]  # (x, y, w, h, angle, score), class is at index 6
            results.append(
                Results(orig_img, path=img_path, names=self.model.names, obb=torch.cat([obb, pred[:, 6:7]], dim=-1))
            )
        return results
