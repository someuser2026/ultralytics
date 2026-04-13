# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import constant_, xavier_uniform_
from torchvision.ops import box_iou, nms


from ultralytics.utils import NOT_MACOS14
from ultralytics.utils.ops import regularize_rboxes
from ultralytics.utils.tal import dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import TORCH_1_11, fuse_conv_and_bn, smart_inference_mode

from .block import DFL, SAVPE, BNContrastiveHead, ContrastiveHead, Proto, Residual, SwiGLUFFN
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, inverse_sigmoid, linear_init

# from .roi_heads import MaskHead, TwoFCBBoxHead, decode_boxes, encode_boxes, roi_align_pyramid
# from .rpn import AnchorGenerator, RPNConfig, RPNHead, rpn_inference_single_image

__all__ = "Detect", "Segment", "Pose", "Classify", "OBB", "RotatedFCOS", "RTDETRDecoder", "RTDETRSegmentDecoder", "RTDETROBBDecoder", "v10Detect", "YOLOEDetect", "YOLOESegment", "Mask2FormerHead" #, "CascadeRCNNHead"


class LearnableScale(nn.Module):
    """A lightweight learnable scalar multiplier."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(init_value)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class FCOSConvModule(nn.Module):
    """Conv-GN-ReLU block used by the Rotated FCOS towers."""

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, 3, padding=1)
        self.norm = nn.GroupNorm(32, c2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Detect(nn.Module):
    """
    YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        forward_end2end: Perform forward pass for end-to-end detection.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """
        Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x: list[torch.Tensor]) -> dict | tuple:
        """
        Perform forward pass of the v10Detect module.

        Args:
            x (list[torch.Tensor]): Input feature maps from different levels.

        Returns:
            outputs (dict | tuple): Training mode returns dict with one2many and one2one outputs.
                Inference mode returns processed detections or tuple with detections and raw outputs.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x: list[torch.Tensor]) -> torch.Tensor:
        """
        Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (list[torch.Tensor]): List of feature maps from different detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides
        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80) -> torch.Tensor:
        """
        Post-process YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class Segment(Detect):
    """
    YOLO Segment head for segmentation models.

    This class extends the Detect head to include mask prediction capabilities for instance segmentation tasks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv4 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a segmentation head
        >>> segment = Segment(nc=80, nm=32, npr=256, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = segment(x)
    """

    def __init__(self, nc: int = 80, nm: int = 32, npr: int = 256, ch: tuple = ()):
        """
        Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> tuple | list[torch.Tensor]:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        # DEBUG (remove after it passes)
        # assert x[0].shape[1] == self.nm, f"Proto nm={self.nm}, got x0 C={x[0].shape[1]}"

        # if not hasattr(self, "_dbg_once"):
        #     print("SEG inputs:", [tuple(t.shape) for t in x], flush=True)
        #     self._dbg_once = True
        
        p = self.proto(x[0])  # mask protos

        # if not hasattr(self, "_dbg_proto"):
        #     print(f"[PROTO] mean={p.mean().item():.4f} std={p.std().item():.4f} shape={tuple(p.shape)}", flush=True)
        #     self._dbg_proto = True
        
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """
    YOLO OBB detection head for detection with rotation models.

    This class extends the Detect head to include oriented bounding box prediction with rotation angles.

    Attributes:
        ne (int): Number of extra parameters.
        cv4 (nn.ModuleList): Convolution layers for angle prediction.
        angle (torch.Tensor): Predicted rotation angles.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        decode_bboxes: Decode rotated bounding boxes.

    Examples:
        Create an OBB detection head
        >>> obb = OBB(nc=80, ne=1, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = obb(x)
    """

    def __init__(self, nc: int = 80, ne: int = 1, ch: tuple = ()):
        """
        Initialize OBB with number of classes `nc` and layer channels `ch`.

        Args:
            nc (int): Number of classes.
            ne (int): Number of extra parameters.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class RotatedFCOS(Detect):
    """MMRotate-style Rotated FCOS head for OBB detection."""

    def __init__(self, nc: int = 80, cfg: dict | None = None, ch: tuple = ()):
        nn.Module.__init__(self)
        cfg = cfg or {}
        if not ch:
            raise ValueError("RotatedFCOS requires at least one input feature map.")
        if len(set(ch)) != 1:
            raise ValueError(f"RotatedFCOS expects equal input channels per level, received {tuple(ch)}.")

        self.nc = nc
        self.nl = len(ch)
        self.no = nc + 5
        self.reg_max = 1
        self.stride = torch.tensor(cfg.get("strides", [8, 16, 32, 64, 128]), dtype=torch.float)
        if len(self.stride) != self.nl:
            raise ValueError(f"RotatedFCOS strides length {len(self.stride)} must match number of levels {self.nl}.")
        self.anchors = torch.empty(0)
        self.strides = torch.empty(0)
        self.shape = None
        self.xyxy = False
        self.max_det = 300
        self.dynamic = False
        self.export = False
        self.format = None
        self.end2end = False
        self.legacy = False

        self.feat_channels = int(cfg.get("feat_channels", 256))
        self.stacked_convs = int(cfg.get("stacked_convs", 4))
        self.regress_ranges = tuple(tuple(r) for r in cfg.get(
            "regress_ranges",
            [(-1, 64), (64, 128), (128, 256), (256, 512), (512, 1e8)],
        ))
        self.center_sampling = bool(cfg.get("center_sampling", False))
        self.center_sample_radius = float(cfg.get("center_sample_radius", 1.5))
        self.norm_on_bbox = bool(cfg.get("norm_on_bbox", False))
        self.centerness_on_reg = bool(cfg.get("centerness_on_reg", False))
        self.scale_angle = bool(cfg.get("scale_angle", True))
        self.bbox_loss_type = str(cfg.get("bbox_loss_type", "rotated_iou"))
        self.angle_mode = str(cfg.get("angle_mode", "oc"))

        c1 = ch[0]
        self.cls_convs = nn.ModuleList(
            FCOSConvModule(c1 if i == 0 else self.feat_channels, self.feat_channels) for i in range(self.stacked_convs)
        )
        self.reg_convs = nn.ModuleList(
            FCOSConvModule(c1 if i == 0 else self.feat_channels, self.feat_channels) for i in range(self.stacked_convs)
        )
        self.conv_cls = nn.Conv2d(self.feat_channels, self.nc, 3, padding=1)
        self.conv_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.conv_angle = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList(LearnableScale(1.0) for _ in range(self.nl))
        self.angle_scale = LearnableScale(1.0) if self.scale_angle else None
        self._init_weights()

    def _init_weights(self):
        """Initialize Rotated FCOS head weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def bias_init(self):
        """Initialize detection biases once strides are available."""
        self.conv_cls.bias.data[: self.nc] = bias_init_with_prob(0.01)
        nn.init.constant_(self.conv_reg.bias, 0)
        nn.init.constant_(self.conv_angle.bias, 0)
        nn.init.constant_(self.conv_centerness.bias, 0)

    def forward_single(self, x: torch.Tensor, scale: LearnableScale, stride: torch.Tensor | float):
        """Forward pass for a single feature level."""
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        cls_score = self.conv_cls(cls_feat)
        centerness = self.conv_centerness(reg_feat if self.centerness_on_reg else cls_feat)
        bbox_pred = scale(self.conv_reg(reg_feat)).float()
        if self.norm_on_bbox:
            bbox_pred = bbox_pred.clamp(min=0)
            if not self.training:
                stride_value = stride.item() if isinstance(stride, torch.Tensor) else float(stride)
                bbox_pred = bbox_pred * stride_value
        else:
            bbox_pred = bbox_pred.exp()
        angle_pred = self.conv_angle(reg_feat)
        if self.angle_scale is not None:
            angle_pred = self.angle_scale(angle_pred).float()
        return cls_score, bbox_pred, angle_pred, centerness

    def _inference(
        self,
        cls_scores: list[torch.Tensor],
        bbox_preds: list[torch.Tensor],
        angle_preds: list[torch.Tensor],
        centernesses: list[torch.Tensor],
    ) -> torch.Tensor:
        """Decode FCOS predictions into Ultralytics rotated prediction format."""
        shape = cls_scores[0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(cls_scores, self.stride, 0.5))
            self.shape = shape

        bs = shape[0]
        cls_logits = torch.cat([x.view(bs, self.nc, -1) for x in cls_scores], 2)
        bbox_dist = torch.cat([x.view(bs, 4, -1) for x in bbox_preds], 2)
        angle = torch.cat([x.view(bs, 1, -1) for x in angle_preds], 2)
        centerness = torch.cat([x.view(bs, 1, -1) for x in centernesses], 2)

        pixel_points = self.anchors.unsqueeze(0) * self.strides
        decoded = dist2rbox(bbox_dist, angle, pixel_points, dim=1)
        rboxes = torch.cat((decoded, angle), 1).transpose(1, 2)
        rboxes = regularize_rboxes(rboxes, angle_mode=self.angle_mode).transpose(1, 2)
        scores = cls_logits.sigmoid() * centerness.sigmoid()
        return torch.cat((rboxes[:, :4], scores, rboxes[:, 4:5]), 1)

    def forward(self, x: list[torch.Tensor]) -> tuple | torch.Tensor:
        """Forward multi-level features through the Rotated FCOS head."""
        outputs = [self.forward_single(feat, scale, stride) for feat, scale, stride in zip(x, self.scales, self.stride)]
        cls_scores, bbox_preds, angle_preds, centernesses = (list(items) for items in zip(*outputs))
        if self.training:
            return cls_scores, bbox_preds, angle_preds, centernesses
        y = self._inference(cls_scores, bbox_preds, angle_preds, centernesses)
        raw = (cls_scores, bbox_preds, angle_preds, centernesses)
        return y if self.export else (y, raw)


class Pose(Detect):
    """
    YOLO Pose head for keypoints models.

    This class extends the Detect head to include keypoint prediction capabilities for pose estimation tasks.

    Attributes:
        kpt_shape (tuple): Number of keypoints and dimensions (2 for x,y or 3 for x,y,visible).
        nk (int): Total number of keypoint values.
        cv4 (nn.ModuleList): Convolution layers for keypoint prediction.

    Methods:
        forward: Perform forward pass through YOLO model and return predictions.
        kpts_decode: Decode keypoints from predictions.

    Examples:
        Create a pose detection head
        >>> pose = Pose(nc=80, kpt_shape=(17, 3), ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = pose(x)
    """

    def __init__(self, nc: int = 80, kpt_shape: tuple = (17, 3), ch: tuple = ()):
        """
        Initialize YOLO network with default parameters and Convolutional Layers.

        Args:
            nc (int): Number of classes.
            kpt_shape (tuple): Number of keypoints, number of dims (2 for x,y or 3 for x,y,visible).
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor | tuple:
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs: int, kpts: torch.Tensor) -> torch.Tensor:
        """Decode keypoints from predictions."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                if NOT_MACOS14:
                    y[:, 2::ndim].sigmoid_()
                else:  # Apple macOS14 MPS bug https://github.com/ultralytics/ultralytics/pull/21878
                    y[:, 2::ndim] = y[:, 2::ndim].sigmoid()
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """
    YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2).

    This class implements a classification head that transforms feature maps into class predictions.

    Attributes:
        export (bool): Export mode flag.
        conv (Conv): Convolutional layer for feature transformation.
        pool (nn.AdaptiveAvgPool2d): Global average pooling layer.
        drop (nn.Dropout): Dropout layer for regularization.
        linear (nn.Linear): Linear layer for final classification.

    Methods:
        forward: Perform forward pass of the YOLO model on input image data.

    Examples:
        Create a classification head
        >>> classify = Classify(c1=1024, c2=1000)
        >>> x = torch.randn(1, 1024, 20, 20)
        >>> output = classify(x)
    """

    export = False  # export mode

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: int | None = None, g: int = 1):
        """
        Initialize YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output classes.
            k (int, optional): Kernel size.
            s (int, optional): Stride.
            p (int, optional): Padding.
            g (int, optional): Groups.
        """
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x: list[torch.Tensor] | torch.Tensor) -> torch.Tensor | tuple:
        """Perform forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """
    Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to incorporate text embeddings for enhanced semantic understanding
    in object detection tasks.

    Attributes:
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.

    Methods:
        forward: Concatenate and return predicted bounding boxes and class probabilities.
        bias_init: Initialize detection head biases.

    Examples:
        Create a WorldDetect head
        >>> world_detect = WorldDetect(nc=80, embed=512, with_bn=False, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = world_detect(x, text)
    """

    def __init__(self, nc: int = 80, embed: int = 512, with_bn: bool = False, ch: tuple = ()):
        """
        Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> list[torch.Tensor] | tuple:
        """Concatenate and return predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class LRPCHead(nn.Module):
    """
    Lightweight Region Proposal and Classification Head for efficient object detection.

    This head combines region proposal filtering with classification to enable efficient detection with
    dynamic vocabulary support.

    Attributes:
        vocab (nn.Module): Vocabulary/classification layer.
        pf (nn.Module): Proposal filter module.
        loc (nn.Module): Localization module.
        enabled (bool): Whether the head is enabled.

    Methods:
        conv2linear: Convert a 1x1 convolutional layer to a linear layer.
        forward: Process classification and localization features to generate detection proposals.

    Examples:
        Create an LRPC head
        >>> vocab = nn.Conv2d(256, 80, 1)
        >>> pf = nn.Conv2d(256, 1, 1)
        >>> loc = nn.Conv2d(256, 4, 1)
        >>> head = LRPCHead(vocab, pf, loc, enabled=True)
    """

    def __init__(self, vocab: nn.Module, pf: nn.Module, loc: nn.Module, enabled: bool = True):
        """
        Initialize LRPCHead with vocabulary, proposal filter, and localization components.

        Args:
            vocab (nn.Module): Vocabulary/classification module.
            pf (nn.Module): Proposal filter module.
            loc (nn.Module): Localization module.
            enabled (bool): Whether to enable the head functionality.
        """
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    def conv2linear(self, conv: nn.Conv2d) -> nn.Linear:
        """Convert a 1x1 convolutional layer to a linear layer."""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.view(conv.out_channels, -1).data
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat: torch.Tensor, loc_feat: torch.Tensor, conf: float) -> tuple[tuple, torch.Tensor]:
        """Process classification and localization features to generate detection proposals."""
        if self.enabled:
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)
            mask = pf_score.sigmoid() > conf
            cls_feat = cls_feat.flatten(2).transpose(-1, -2)
            cls_feat = self.vocab(cls_feat[:, mask] if conf else cls_feat * mask.unsqueeze(-1).int())
            return (self.loc(loc_feat), cls_feat.transpose(-1, -2)), mask
        else:
            cls_feat = self.vocab(cls_feat)
            loc_feat = self.loc(loc_feat)
            return (loc_feat, cls_feat.flatten(2)), torch.ones(
                cls_feat.shape[2] * cls_feat.shape[3], device=cls_feat.device, dtype=torch.bool
            )


class YOLOEDetect(Detect):
    """
    Head for integrating YOLO detection models with semantic understanding from text embeddings.

    This class extends the standard Detect head to support text-guided detection with enhanced semantic understanding
    through text embeddings and visual prompt embeddings.

    Attributes:
        is_fused (bool): Whether the model is fused for inference.
        cv3 (nn.ModuleList): Convolution layers for embedding features.
        cv4 (nn.ModuleList): Contrastive head layers for text-vision alignment.
        reprta (Residual): Residual block for text prompt embeddings.
        savpe (SAVPE): Spatial-aware visual prompt embeddings module.
        embed (int): Embedding dimension.

    Methods:
        fuse: Fuse text features with model weights for efficient inference.
        get_tpe: Get text prompt embeddings with normalization.
        get_vpe: Get visual prompt embeddings with spatial awareness.
        forward_lrpc: Process features with fused text embeddings for prompt-free model.
        forward: Process features with class prompt embeddings to generate detections.
        bias_init: Initialize biases for detection heads.

    Examples:
        Create a YOLOEDetect head
        >>> yoloe_detect = YOLOEDetect(nc=80, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> cls_pe = torch.randn(1, 80, 512)
        >>> outputs = yoloe_detect(x, cls_pe)
    """

    is_fused = False

    def __init__(self, nc: int = 80, embed: int = 512, with_bn: bool = False, ch: tuple = ()):
        """
        Initialize YOLO detection layer with nc classes and layer channels ch.

        Args:
            nc (int): Number of classes.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        assert c3 <= embed
        assert with_bn
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, embed, 1),
                )
                for x in ch
            )
        )

        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)
        self.embed = embed

    @smart_inference_mode()
    def fuse(self, txt_feats: torch.Tensor):
        """Fuse text features with model weights for efficient inference."""
        if self.is_fused:
            return

        assert not self.training
        txt_feats = txt_feats.to(torch.float32).squeeze(0)
        for cls_head, bn_head in zip(self.cv3, self.cv4):
            assert isinstance(cls_head, nn.Sequential)
            assert isinstance(bn_head, BNContrastiveHead)
            conv = cls_head[-1]
            assert isinstance(conv, nn.Conv2d)
            logit_scale = bn_head.logit_scale
            bias = bn_head.bias
            norm = bn_head.norm

            t = txt_feats * logit_scale.exp()
            conv: nn.Conv2d = fuse_conv_and_bn(conv, norm)

            w = conv.weight.data.squeeze(-1).squeeze(-1)
            b = conv.bias.data

            w = t @ w
            b1 = (t @ b.reshape(-1).unsqueeze(-1)).squeeze(-1)
            b2 = torch.ones_like(b1) * bias

            conv = (
                nn.Conv2d(
                    conv.in_channels,
                    w.shape[0],
                    kernel_size=1,
                )
                .requires_grad_(False)
                .to(conv.weight.device)
            )

            conv.weight.data.copy_(w.unsqueeze(-1).unsqueeze(-1))
            conv.bias.data.copy_(b1 + b2)
            cls_head[-1] = conv

            bn_head.fuse()

        del self.reprta
        self.reprta = nn.Identity()
        self.is_fused = True

    def get_tpe(self, tpe: torch.Tensor | None) -> torch.Tensor | None:
        """Get text prompt embeddings with normalization."""
        return None if tpe is None else F.normalize(self.reprta(tpe), dim=-1, p=2)

    def get_vpe(self, x: list[torch.Tensor], vpe: torch.Tensor) -> torch.Tensor:
        """Get visual prompt embeddings with spatial awareness."""
        if vpe.shape[1] == 0:  # no visual prompt embeddings
            return torch.zeros(x[0].shape[0], 0, self.embed, device=x[0].device)
        if vpe.ndim == 4:  # (B, N, H, W)
            vpe = self.savpe(x, vpe)
        assert vpe.ndim == 3  # (B, N, D)
        return vpe

    def forward_lrpc(self, x: list[torch.Tensor], return_mask: bool = False) -> torch.Tensor | tuple:
        """Process features with fused text embeddings to generate detections for prompt-free model."""
        masks = []
        assert self.is_fused, "Prompt-free inference requires model to be fused!"
        for i in range(self.nl):
            cls_feat = self.cv3[i](x[i])
            loc_feat = self.cv2[i](x[i])
            assert isinstance(self.lrpc[i], LRPCHead)
            x[i], mask = self.lrpc[i](
                cls_feat, loc_feat, 0 if self.export and not self.dynamic else getattr(self, "conf", 0.001)
            )
            masks.append(mask)
        shape = x[0][0].shape
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors([b[0] for b in x], self.stride, 0.5))
            self.shape = shape
        box = torch.cat([xi[0].view(shape[0], self.reg_max * 4, -1) for xi in x], 2)
        cls = torch.cat([xi[1] for xi in x], 2)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        mask = torch.cat(masks)
        y = torch.cat((dbox if self.export and not self.dynamic else dbox[..., mask], cls.sigmoid()), 1)

        if return_mask:
            return (y, mask) if self.export else ((y, x), mask)
        else:
            return y if self.export else (y, x)

    def forward(self, x: list[torch.Tensor], cls_pe: torch.Tensor, return_mask: bool = False) -> torch.Tensor | tuple:
        """Process features with class prompt embeddings to generate detections."""
        if hasattr(self, "lrpc"):  # for prompt-free inference
            return self.forward_lrpc(x, return_mask)
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), cls_pe)), 1)
        if self.training:
            return x
        self.no = self.nc + self.reg_max * 4  # self.nc could be changed when inference with different texts
        y = self._inference(x)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize biases for detection heads."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, c, s in zip(m.cv2, m.cv3, m.cv4, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
            b[-1].bias.data[:] = 0.0
            c.bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)


class YOLOESegment(YOLOEDetect):
    """
    YOLO segmentation head with text embedding capabilities.

    This class extends YOLOEDetect to include mask prediction capabilities for instance segmentation tasks
    with text-guided semantic understanding.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of protos.
        proto (Proto): Prototype generation module.
        cv5 (nn.ModuleList): Convolution layers for mask coefficients.

    Methods:
        forward: Return model outputs and mask coefficients.

    Examples:
        Create a YOLOESegment head
        >>> yoloe_segment = YOLOESegment(nc=80, nm=32, npr=256, embed=512, with_bn=True, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> text = torch.randn(1, 80, 512)
        >>> outputs = yoloe_segment(x, text)
    """

    def __init__(
        self, nc: int = 80, nm: int = 32, npr: int = 256, embed: int = 512, with_bn: bool = False, ch: tuple = ()
    ):
        """
        Initialize YOLOESegment with class count, mask parameters, and embedding dimensions.

        Args:
            nc (int): Number of classes.
            nm (int): Number of masks.
            npr (int): Number of protos.
            embed (int): Embedding dimension.
            with_bn (bool): Whether to use batch normalization in contrastive head.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, embed, with_bn, ch)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], self.npr, self.nm)

        c5 = max(ch[0] // 4, self.nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, self.nm, 1)) for x in ch)

    def forward(self, x: list[torch.Tensor], text: torch.Tensor) -> tuple | torch.Tensor:
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv5[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        has_lrpc = hasattr(self, "lrpc")

        if not has_lrpc:
            x = YOLOEDetect.forward(self, x, text)
        else:
            x, mask = YOLOEDetect.forward(self, x, text, return_mask=True)

        if self.training:
            return x, mc, p

        if has_lrpc:
            mc = (mc * mask.int()) if self.export and not self.dynamic else mc[..., mask]

        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.

    Attributes:
        export (bool): Export mode flag.
        hidden_dim (int): Dimension of hidden layers.
        nhead (int): Number of heads in multi-head attention.
        nl (int): Number of feature levels.
        nc (int): Number of classes.
        num_queries (int): Number of query points.
        num_decoder_layers (int): Number of decoder layers.
        input_proj (nn.ModuleList): Input projection layers for backbone features.
        decoder (DeformableTransformerDecoder): Transformer decoder module.
        denoising_class_embed (nn.Embedding): Class embeddings for denoising.
        num_denoising (int): Number of denoising queries.
        label_noise_ratio (float): Label noise ratio for training.
        box_noise_scale (float): Box noise scale for training.
        learnt_init_query (bool): Whether to learn initial query embeddings.
        tgt_embed (nn.Embedding): Target embeddings for queries.
        query_pos_head (MLP): Query position head.
        enc_output (nn.Sequential): Encoder output layers.
        enc_score_head (nn.Linear): Encoder score prediction head.
        enc_bbox_head (MLP): Encoder bbox prediction head.
        dec_score_head (nn.ModuleList): Decoder score prediction heads.
        dec_bbox_head (nn.ModuleList): Decoder bbox prediction heads.

    Methods:
        forward: Run forward pass and return bounding box and classification scores.

    Examples:
        Create an RTDETRDecoder
        >>> decoder = RTDETRDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    export = False  # export mode

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,  # hidden dim
        nq: int = 300,  # num queries
        ndp: int = 4,  # num decoder points
        nh: int = 8,  # num head
        ndl: int = 6,  # num decoder layers
        d_ffn: int = 1024,  # dim of feedforward
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        # Training args
        nd: int = 100,  # num denoising
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """
        Initialize the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Dimension of hidden layers.
            nq (int): Number of query points.
            ndp (int): Number of decoder points.
            nh (int): Number of heads in multi-head attention.
            ndl (int): Number of decoder layers.
            d_ffn (int): Dimension of the feed-forward networks.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """
        Run the forward pass of the module, returning bounding box and classification scores for the input.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            outputs (tuple | torch.Tensor): During training, returns a tuple of bounding boxes, scores, and other
                metadata. During inference, returns a tensor of shape (bs, 300, 4+nc) containing bounding boxes and
                class scores.
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(
        self,
        shapes: list[list[int]],
        grid_size: float = 0.05,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        eps: float = 1e-2,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate anchor bounding boxes for given shapes with specific grid size and validate them.

        Args:
            shapes (list): List of feature map shapes.
            grid_size (float, optional): Base size of grid cells.
            dtype (torch.dtype, optional): Data type for tensors.
            device (str, optional): Device to create tensors on.
            eps (float, optional): Small value for numerical stability.

        Returns:
            anchors (torch.Tensor): Generated anchor boxes.
            valid_mask (torch.Tensor): Valid mask for anchors.
        """
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_11 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x: list[torch.Tensor]) -> tuple[torch.Tensor, list[list[int]]]:
        """
        Process and return encoder inputs by getting projection features from input and concatenating them.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.

        Returns:
            feats (torch.Tensor): Processed features.
            shapes (list): List of feature map shapes.
        """
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate and prepare the input required for the decoder from the provided features and shapes.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes.

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes.
            enc_bboxes (torch.Tensor): Encoded bounding boxes.
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def _reset_parameters(self):
        """Initialize or reset the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class RTDETRSegmentDecoder(RTDETRDecoder):
    """
    Real-Time Deformable Transformer Decoder for Segmentation tasks.

    This decoder extends RTDETRDecoder to predict both bounding boxes and segmentation masks.

    Attributes:
        nm (int): Number of masks.
        npr (int): Number of prototypes.
        proto (Proto): Prototype generation module for masks.
        enc_mask_head (nn.Linear): Encoder mask coefficient head.
        dec_mask_head (nn.ModuleList): Decoder mask coefficient heads.

    Examples:
        Create an RTDETRSegmentDecoder
        >>> decoder = RTDETRSegmentDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300, nm=32, npr=256)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
        nm: int = 32,  # number of masks
        npr: int = 256,  # number of protos
    ):
        """
        Initialize RTDETRSegmentDecoder with segmentation-specific parameters.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Hidden dimension.
            nq (int): Number of queries.
            ndp (int): Number of decoder points.
            nh (int): Number of heads.
            ndl (int): Number of decoder layers.
            d_ffn (int): Feed-forward dimension.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising queries.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
            nm (int): Number of masks.
            npr (int): Number of prototypes.
        """
        super().__init__(
            nc=nc,
            ch=ch,
            hd=hd,
            nq=nq,
            ndp=ndp,
            nh=nh,
            ndl=ndl,
            d_ffn=d_ffn,
            dropout=dropout,
            act=act,
            eval_idx=eval_idx,
            nd=nd,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learnt_init_query=learnt_init_query,
        )
        self.nm = nm
        self.npr = npr
        # Proto module for mask generation
        self.proto = Proto(ch[0], self.npr, self.nm)

        # Mask prediction heads
        self.enc_mask_head = nn.Linear(hd, nm)
        self.dec_mask_head = nn.ModuleList([nn.Linear(hd, nm) for _ in range(ndl)])

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """
        Run forward pass returning bounding boxes, scores, mask coefficients, and prototypes.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            During training: (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dec_masks, enc_masks, protos, dn_meta)
            During inference: (y, (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dec_masks, enc_masks, protos, dn_meta))
                where y is (bs, 300, 4+nc+nm) concatenated tensor
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Generate prototypes from first feature map
        protos = self.proto(x[0])  # (bs, nm, H, W)

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Encoder mask predictions - use same top-k selection as encoder scores
        _, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        enc_features = self.enc_output(valid_mask * feats)  # (bs, h*w, hd)
        enc_mask_coeffs_full = self.enc_mask_head(enc_features)  # (bs, h*w, nm)
        # Get top-k indices from encoder scores (from _get_decoder_input)
        enc_outputs_scores_full = self.enc_score_head(enc_features)
        bs = enc_features.shape[0]
        topk_ind = torch.topk(enc_outputs_scores_full.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)
        enc_mask_coeffs = enc_mask_coeffs_full[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        # Decoder - manually run to track mask predictions at each layer
        output = embed
        dec_mask_coeffs = []
        refer_bbox_detached = refer_bbox.sigmoid() if not self.training else refer_bbox.detach().sigmoid()
        last_refined_bbox = None
        dec_bboxes_list = []
        dec_scores_list = []

        for i, layer in enumerate(self.decoder.layers):
            output = layer(
                output,
                refer_bbox_detached,
                feats,
                shapes,
                None,  # padding_mask
                attn_mask,
                self.query_pos_head(refer_bbox_detached),
            )
            
            # Predict bbox, score, and mask for this layer
            bbox = self.dec_bbox_head[i](output)
            refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(refer_bbox_detached))
            if i > 0:
                refined_bbox = torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox))
            
            score = self.dec_score_head[i](output)
            mask_coeff = self.dec_mask_head[i](output)
            dec_mask_coeffs.append(mask_coeff)
            
            if self.training:
                dec_scores_list.append(score)
                if i == 0:
                    dec_bboxes_list.append(refined_bbox)
                else:
                    dec_bboxes_list.append(torch.sigmoid(bbox + inverse_sigmoid(last_refined_bbox)))
            elif i == self.decoder.eval_idx:
                # For inference, stop at eval_idx - use current predictions
                dec_bboxes = refined_bbox.unsqueeze(0)  # (1, bs, nq, 4)
                dec_scores = score.unsqueeze(0)  # (1, bs, nq, nc)
                dec_mask_coeffs = mask_coeff.unsqueeze(0)  # (1, bs, nq, nm)
                break

            last_refined_bbox = refined_bbox
            refer_bbox_detached = refined_bbox.detach() if self.training else refined_bbox

        # Stack decoder outputs for training
        if self.training:
            dec_bboxes = torch.stack(dec_bboxes_list)  # (ndl, bs, nq, 4)
            dec_scores = torch.stack(dec_scores_list)  # (ndl, bs, nq, nc)
            dec_mask_coeffs = torch.stack(dec_mask_coeffs)  # (ndl, bs, nq, nm)

        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dec_mask_coeffs, enc_mask_coeffs, protos, dn_meta
        if self.training:
            return x
        # Concatenate bboxes, scores, and mask coefficients for inference
        # dec_bboxes, dec_scores, dec_mask_coeffs are already at eval_idx from the loop above
        dec_bboxes_eval = dec_bboxes.squeeze(0)  # (bs, nq, 4)
        dec_scores_eval = dec_scores.squeeze(0)  # (bs, nq, nc)
        dec_masks_eval = dec_mask_coeffs.squeeze(0)  # (bs, nq, nm)

        # (bs, 300, 4+nc+nm)
        y = torch.cat((dec_bboxes_eval, dec_scores_eval.sigmoid(), dec_masks_eval), -1)
        return (y, protos) if self.export else (y, x)



class RTDETROBBDecoder(RTDETRDecoder):
    """
    Real-Time Deformable Transformer Decoder for Oriented Bounding Box tasks.

    This decoder extends RTDETRDecoder to predict rotated bounding boxes with angles.

    Attributes:
        enc_bbox_head (MLP): Encoder bbox head outputting 5D (x, y, w, h, angle).
        dec_bbox_head (nn.ModuleList): Decoder bbox heads outputting 5D.

    Examples:
        Create an RTDETROBBDecoder
        >>> decoder = RTDETROBBDecoder(nc=80, ch=(512, 1024, 2048), hd=256, nq=300)
        >>> x = [torch.randn(1, 512, 64, 64), torch.randn(1, 1024, 32, 32), torch.randn(1, 2048, 16, 16)]
        >>> outputs = decoder(x)
    """

    def __init__(
        self,
        nc: int = 80,
        ch: tuple = (512, 1024, 2048),
        hd: int = 256,
        nq: int = 300,
        ndp: int = 4,
        nh: int = 8,
        ndl: int = 6,
        d_ffn: int = 1024,
        dropout: float = 0.0,
        act: nn.Module = nn.ReLU(),
        eval_idx: int = -1,
        nd: int = 100,
        label_noise_ratio: float = 0.5,
        box_noise_scale: float = 1.0,
        learnt_init_query: bool = False,
    ):
        """
        Initialize RTDETROBBDecoder for oriented bounding box detection.

        Args:
            nc (int): Number of classes.
            ch (tuple): Channels in the backbone feature maps.
            hd (int): Hidden dimension.
            nq (int): Number of queries.
            ndp (int): Number of decoder points.
            nh (int): Number of heads.
            ndl (int): Number of decoder layers.
            d_ffn (int): Feed-forward dimension.
            dropout (float): Dropout rate.
            act (nn.Module): Activation function.
            eval_idx (int): Evaluation index.
            nd (int): Number of denoising queries.
            label_noise_ratio (float): Label noise ratio.
            box_noise_scale (float): Box noise scale.
            learnt_init_query (bool): Whether to learn initial query embeddings.
        """
        super().__init__(
            nc=nc,
            ch=ch,
            hd=hd,
            nq=nq,
            ndp=ndp,
            nh=nh,
            ndl=ndl,
            d_ffn=d_ffn,
            dropout=dropout,
            act=act,
            eval_idx=eval_idx,
            nd=nd,
            label_noise_ratio=label_noise_ratio,
            box_noise_scale=box_noise_scale,
            learnt_init_query=learnt_init_query,
        )

        # Override bbox heads to output 5D (x, y, w, h, angle) instead of 4D
        self.enc_bbox_head = MLP(hd, hd, 5, num_layers=3)
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 5, num_layers=3) for _ in range(ndl)])

        self.query_pos_head = MLP(5, 2 * hd, hd, num_layers=2)
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp, use_obb=True)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Re-initialize bbox head parameters
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for reg_ in self.dec_bbox_head:
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

    def _get_decoder_input_obb(
        self,
        feats: torch.Tensor,
        shapes: list[list[int]],
        dn_embed: torch.Tensor | None = None,
        dn_bbox: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate and prepare the input required for the OBB decoder from the provided features and shapes.
        Handles 5D bboxes (x, y, w, h, angle) instead of 4D.

        Args:
            feats (torch.Tensor): Processed features from encoder.
            shapes (list): List of feature map shapes.
            dn_embed (torch.Tensor, optional): Denoising embeddings.
            dn_bbox (torch.Tensor, optional): Denoising bounding boxes (may be 5D).

        Returns:
            embeddings (torch.Tensor): Query embeddings for decoder.
            refer_bbox (torch.Tensor): Reference bounding boxes (5D).
            enc_bboxes (torch.Tensor): Encoded bounding boxes (5D).
            enc_scores (torch.Tensor): Encoded scores.
        """
        bs = feats.shape[0]
        # Prepare input for decoder - generate 4D anchors first
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)

        enc_outputs_scores = self.enc_score_head(features)

        # Query selection
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)  # (bs, nq, 4)

        # Dynamic anchors + static content - use a zero-angle prior in normalized angle space.
        enc_bbox_pred = self.enc_bbox_head(top_k_features)  # (bs, nq, 5)
        zero_angle = torch.full_like(top_k_anchors[:, :, :1], math.log(0.25 / 0.75))
        top_k_anchors_5d = torch.cat([top_k_anchors, zero_angle], dim=-1)  # (bs, nq, 5)
        refer_bbox = enc_bbox_pred + top_k_anchors_5d

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            # dn_bbox may be 4D or 5D - pad if needed
            if dn_bbox.shape[-1] == 4:
                dn_zero_angle = torch.full_like(dn_bbox[:, :, :1], math.log(0.25 / 0.75))
                dn_bbox_5d = torch.cat([dn_bbox, dn_zero_angle], dim=-1)
                refer_bbox = torch.cat([dn_bbox_5d, refer_bbox], 1)
            else:
                refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    def forward(self, x: list[torch.Tensor], batch: dict | None = None) -> tuple | torch.Tensor:
        """
        Run forward pass returning rotated bounding boxes (5D) and classification scores.

        Args:
            x (list[torch.Tensor]): List of feature maps from the backbone.
            batch (dict, optional): Batch information for training.

        Returns:
            During training: (dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta)
                where bboxes are 5D (x, y, w, h, angle) with angle in [-pi/4, 3pi/4]
            During inference: (y, x) where y is (bs, 300, 5+nc) concatenated tensor
        """
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input_obb(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )

        # Regularize normalized angles to [-pi/4, 3pi/4].
        def regularize_angle(angle_tensor):
            """Regularize normalized angle from [0, 1] to [-pi/4, 3pi/4]."""
            return (angle_tensor - 0.25) * math.pi

        if enc_bboxes.shape[-1] == 5:
            enc_bboxes = torch.cat([enc_bboxes[..., :4], regularize_angle(enc_bboxes[..., 4:5])], dim=-1)

        if dec_bboxes.shape[-1] == 5:
            dec_bboxes = torch.cat([dec_bboxes[..., :4], regularize_angle(dec_bboxes[..., 4:5])], dim=-1)

        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 5+nc)
        dec_bboxes_eval = dec_bboxes.squeeze(0)
        dec_scores_eval = dec_scores.squeeze(0)
        y = torch.cat((dec_bboxes_eval, dec_scores_eval.sigmoid()), -1)
        return y if self.export else (y, x)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    This class implements the YOLOv10 detection head with dual-assignment training and consistent dual predictions
    for improved efficiency and performance.

    Attributes:
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum number of detections.
        cv3 (nn.ModuleList): Light classification head layers.
        one2one_cv3 (nn.ModuleList): One-to-one classification head layers.

    Methods:
        __init__: Initialize the v10Detect object with specified number of classes and input channels.
        forward: Perform forward pass of the v10Detect module.
        bias_init: Initialize biases of the Detect module.
        fuse: Remove the one2many head for inference optimization.

    Examples:
        Create a v10Detect head
        >>> v10_detect = v10Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = v10_detect(x)
    """

    end2end = True

    def __init__(self, nc: int = 80, ch: tuple = ()):
        """
        Initialize the v10Detect object with the specified number of classes and input channels.

        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def fuse(self):
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = nn.ModuleList([nn.Identity()] * self.nl)

# # from .roi_heads import MaskHead, TwoFCBBoxHead, decode_boxes, encode_boxes, roi_align_pyramid
# # from .rpn import AnchorGenerator, RPNConfig, RPNHead, rpn_inference_single_image

# def _assign_samples(
#     proposals: torch.Tensor,
#     gt_boxes: torch.Tensor,
#     gt_classes: torch.Tensor,
#     iou_thr_pos: float,
#     iou_thr_neg: float,
#     samples_per_img: int,
#     fg_fraction: float,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Assign proposals to GT and sample for training.
    
#     Returns:
#         sampled_inds: Indices in proposals [Ni]
#         labels: -1=ignore, 0=bg, >0=class_id [Ni]
#         matched_gt: GT boxes for positives [Ni, 4]
#         matched_cls: Class ids [Ni]
#     """
#     if gt_boxes.numel() == 0 or proposals.numel() == 0:
#         return (
#             torch.zeros(0, dtype=torch.long, device=proposals.device),
#             torch.zeros(0, dtype=torch.long, device=proposals.device),
#             torch.zeros((0, 4), device=proposals.device),
#             torch.zeros(0, dtype=torch.long, device=proposals.device),
#         )

#     ious = box_iou(proposals, gt_boxes)
#     iou_vals, gt_idx = ious.max(dim=1)

#     labels = torch.full((proposals.shape[0],), -1, dtype=torch.long, device=proposals.device)
#     labels[iou_vals < iou_thr_neg] = 0
#     pos = iou_vals >= iou_thr_pos
#     labels[pos] = gt_classes[gt_idx[pos]]

#     # Sample positives and negatives
#     num_samples = min(samples_per_img, proposals.shape[0])
#     num_pos = int(fg_fraction * num_samples)
#     pos_idx = torch.nonzero(labels > 0, as_tuple=False).flatten()
#     neg_idx = torch.nonzero(labels == 0, as_tuple=False).flatten()

#     if pos_idx.numel() > num_pos:
#         perm = torch.randperm(pos_idx.numel(), device=proposals.device)[:num_pos]
#         pos_idx = pos_idx[perm]
#     if neg_idx.numel() > (num_samples - pos_idx.numel()):
#         perm = torch.randperm(neg_idx.numel(), device=proposals.device)[: (num_samples - pos_idx.numel())]
#         neg_idx = neg_idx[perm]

#     sampled_inds = torch.cat([pos_idx, neg_idx], dim=0)
#     matched_gt = gt_boxes[gt_idx[sampled_inds].clamp(min=0)]
#     matched_cls = torch.clamp(labels[sampled_inds], min=0)
#     return sampled_inds, labels[sampled_inds], matched_gt, matched_cls


# class CascadeRCNNHead(nn.Module):
#     """
#     Cascade R-CNN head with RPN and multi-stage refinement.
    
#     Supports both detection and instance segmentation.
#     """

#     def __init__(
#         self,
#         in_channels: List[int],
#         nc: int,
#         cfg: dict,
#         strides: List[int] | None = None,
#     ):
#         """
#         Initialize Cascade R-CNN head.
        
#         Args:
#             in_channels: FPN feature channels per level
#             nc: Number of classes
#             rpn_cfg: RPN configuration dict
#             roi_cfg: ROI pooling configuration
#             cas_cfg: Cascade configuration
#             mask_cfg: Mask head configuration (optional)
#             strides: Feature map strides
#         """
#         super().__init__()
#         self.nc = int(nc)
#         self.strides = strides or [8, 16, 32, 64]
#         print(cfg)

#         # RPN
#         rpn_cfg = cfg["rpn"]
#         self.rpn_cfg = RPNConfig(**rpn_cfg)
#         num_anchors = len(self.rpn_cfg.ratios)
#         self.rpn_head = nn.ModuleList([RPNHead(c, num_anchors) for c in in_channels])
#         self.anchor_gen = AnchorGenerator(self.rpn_cfg.anchor_sizes, self.rpn_cfg.ratios, self.strides)

#         # ROI heads
#         roi_cfg = cfg["roi"]
#         self.pooler_resolution = int(roi_cfg.get("pooler_resolution", 7))
#         self.pooler_sampling = int(roi_cfg.get("pooler_sampling", 2))
        
#         cas_cfg = cfg["cas"]
#         stages = int(cas_cfg.get("stages", 3))
#         iou_thr = cas_cfg.get("iou_thr", [0.5, 0.6, 0.7])
#         bbox_std = cas_cfg.get("bbox_std", [[0.1, 0.1, 0.2, 0.2], [0.05, 0.05, 0.1, 0.1], [0.033, 0.033, 0.067, 0.067]])
        
#         self.stage_iou = [float(x) for x in iou_thr][:stages]
#         self.stage_std = [tuple(map(float, s)) for s in bbox_std][:stages]
#         self.stage_heads = nn.ModuleList([TwoFCBBoxHead(in_channels[0], self.pooler_resolution, self.nc) for _ in range(stages)])

#         # Mask head (optional)
#         mask_cfg = cfg["mask"]
#         self.with_mask = bool(mask_cfg and mask_cfg.get("with_mask", False))
#         self.mask_size = int(mask_cfg.get("mask_size", 28)) if mask_cfg else 28
#         if self.with_mask:
#             self.mask_pool_res = max(14, self.pooler_resolution * 2)
#             self.mask_head = MaskHead(in_channels[0], self.nc, mask_size=self.mask_size)

#     def _get_name(self):
#         """Return model name for YOLO rerouting."""
#         return "CascadeRCNNHead"

#     @staticmethod
#     def _level_assign(boxes: torch.Tensor, strides: List[int]) -> List[int]:
#         """Assign ROIs to FPN levels based on box size."""
#         if boxes.numel() == 0:
#             return []
#         ws = boxes[:, 2] - boxes[:, 0]
#         hs = boxes[:, 3] - boxes[:, 1]
#         s = torch.sqrt(torch.clamp(ws * hs, min=1.0))
#         lvl = torch.clamp(((s / 224.0).log2() * 4.0 + 4.0).round().long(), min=0, max=len(strides) - 1)
#         return lvl.tolist()

#     def forward(
#         self,
#         feats: List[torch.Tensor]
#     ):
#         """
#         Forward pass.
        
#         Training returns dict for loss computation.
#         Inference returns raw predictions
#         """
#         B = feats[0].shape[0]
#         # H, W = imgsz if imgsz is not None else (feats[0].shape[-2], feats[0].shape[-1])

#         # RPN forward
#         rpn_logits_per_level, rpn_deltas_per_level = [], []
#         for l, head in enumerate(self.rpn_head):
#             lo, dr = head([feats[l]])
#             rpn_logits_per_level.append(lo[0])
#             rpn_deltas_per_level.append(dr[0])
        
#         anchors_per_level = self.anchor_gen(feats)
#         if self.training:
#             return {
#                 "rpn_logits": rpn_logits_per_level,
#                 "rpn_deltas": rpn_deltas_per_level,
#                 "anchors": anchors_per_level,
#                 "feats": feats,
#                 "pooler_resolution": self.pooler_resolution,
#                 "pooler_samplinng": self.pooler_sampling,
#                 "stage_std": self.stage_std,
#                 "stage_iou": self.stage_iou,
#             }
        
#         H = int(feats[0].shape[-2] * self.stride[0])
#         W = int(feats[0].shape[-1] * self.stride[0])

#         proposals = []
#         for i in range(B):
#             props_i = rpn_inference_single_image(
#                 [x[i] for x in rpn_logits_per_level],
#                 [x[i] for x in rpn_deltas_per_level],
#                 anchors_per_level,
#                 (H, W),
#                 self.rpn_cfg,
#             )
#             proposals.append(props_i[:, :4])

#         # Cascade stages
#         cascade_out = []
#         proposals_s = proposals

#         for s, head in enumerate(self.stage_heads):
#             std = self.stage_std[s]
            
#             # Assign levels and pool
#             levels_all = []
#             rois_all = []
#             for i in range(B):
#                 pi = proposals_s[i]
#                 rois_all.append(pi)
#                 levels_all += self._level_assign(pi, self.strides)
            
#             pooled = roi_align_pyramid(feats, rois_all, levels_all, self.pooler_resolution, self.pooler_sampling)

#             # Split by image
#             counts = [p.shape[0] for p in proposals_s]
#             if sum(counts) == 0:
#                 cascade_out.append(
#                     {"cls_logits": torch.zeros((0, self.nc), device=feats[0].device), "bbox_deltas": torch.zeros((0, 4 * self.nc), device=feats[0].device)}
#                 )
#                 continue

#             splits = torch.split(pooled, counts, dim=0)
#             out_logits, out_deltas = [], []
#             for x in splits:
#                 if x.numel() == 0:
#                     out_logits.append(torch.zeros((0, self.nc), device=x.device))
#                     out_deltas.append(torch.zeros((0, 4 * self.nc), device=x.device))
#                     continue
#                 out = head(x)
#                 out_logits.append(out["cls_logits"])
#                 out_deltas.append(out["bbox_deltas"])

#             cls_logits = torch.cat(out_logits, dim=0)
#             bbox_deltas = torch.cat(out_deltas, dim=0)
#             cascade_out.append({"cls_logits": cls_logits, "bbox_deltas": bbox_deltas})

#             # Refine proposals for next stage
#             start = 0
#             new_props = []
#             for i, n in enumerate(counts):
#                 if n == 0:
#                     new_props.append(proposals_s[i])
#                     continue
#                 end = start + n
#                 pl = proposals_s[i]
#                 logit_i = cls_logits[start:end]
#                 delta_i = bbox_deltas[start:end]
                
#                 cls_ids = logit_i.argmax(dim=1).clamp(min=0)
#                 idx = cls_ids[:, None] * 4 + torch.tensor([0, 1, 2, 3], device=delta_i.device)[None, :]
#                 deltas_sel = torch.gather(delta_i, 1, idx)
#                 boxes_ref = decode_boxes(pl, deltas_sel, std)
                
#                 boxes_ref[:, [0, 2]] = boxes_ref[:, [0, 2]].clamp(0, W - 1)
#                 boxes_ref[:, [1, 3]] = boxes_ref[:, [1, 3]].clamp(0, H - 1)
#                 new_props.append(boxes_ref)
#                 start = end
#             proposals_s = new_props

#         # Inference: convert last stage to results
#         last = cascade_out[-1]
#         start = 0
#         results = []
#         for i, props in enumerate(proposals_s):
#             n = props.shape[0]
#             if n == 0:
#                 results.append(
#                     {
#                         "bboxes": torch.zeros((0, 4), device=feats[0].device),
#                         "conf": torch.zeros((0,), device=feats[0].device),
#                         "cls": torch.zeros((0,), dtype=torch.long, device=feats[0].device),
#                     }
#                 )
#                 continue

#             end = start + n
#             logits_i = last["cls_logits"][start:end]
#             deltas_i = last["bbox_deltas"][start:end]
            
#             probs = logits_i.softmax(dim=1)
#             scores, labels = probs.max(dim=1)
            
#             idx = labels[:, None] * 4 + torch.tensor([0, 1, 2, 3], device=deltas_i.device)[None, :]
#             deltas_sel = torch.gather(deltas_i, 1, idx)
#             boxes = decode_boxes(props, deltas_sel, self.stage_std[-1])
            
#             boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, W - 1)
#             boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, H - 1)
            
#             # Per-class NMS
#             keep_all = []
#             for c in range(self.nc):
#                 m = (labels == c).nonzero(as_tuple=False).flatten()
#                 if m.numel() == 0:
#                     continue
#                 keep_c = nms(boxes[m], scores[m], 0.5)
#                 keep_all.append(m[keep_c])
            
#             keep = torch.cat(keep_all, dim=0) if keep_all else torch.zeros((0,), dtype=torch.long, device=feats[0].device)
#             results.append({"bboxes": boxes[keep], "conf": scores[keep], "cls": labels[keep]})
#             start = end

#         return results

# --- Mask2Former-like Head (ULY) --------------------------------------------
from .transformer import MSDeformAttn, MLP  # reuse your existing modules

# __all__ = (*__all__, "Mask2FormerHeadULY") if isinstance(__all__, tuple) else "Mask2FormerHeadULY"


class _SinePosEnc2D(nn.Module):
    """2D sine-cosine PE: gives geometry to attention."""
    def __init__(self, dim: int = 256, temperature: int = 10000):
        super().__init__()
        assert dim % 2 == 0
        self.half = dim // 2
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate 2D sine-cosine positional encodings for spatial features.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Positional encoding tensor of shape (B, dim, H, W)
        """
        b, _, h, w = x.shape
        device, dtype = x.device, x.dtype
        
        # Create normalized y and x coordinate grids in range [0, h-1] and [0, w-1]
        yy = torch.linspace(0, h - 1, h, device=device, dtype=dtype).unsqueeze(1).repeat(1, w)
        xx = torch.linspace(0, w - 1, w, device=device, dtype=dtype).unsqueeze(0).repeat(h, 1)
        
        # Compute temperature-based dimension scaling for positional encoding
        dim_t = self.temperature ** (2 * (torch.arange(self.half, device=device, dtype=dtype) // 2) / self.half)
        
        # Apply sine and cosine functions to x and y coordinates
        px = xx[..., None] / dim_t
        py = yy[..., None] / dim_t
        px = torch.stack((px.sin(), px.cos()), dim=-1).flatten(-2)
        py = torch.stack((py.sin(), py.cos()), dim=-1).flatten(-2)
        
        # Concatenate y and x encodings and reshape to (C, H, W)
        pos = torch.cat((py, px), dim=-1).permute(2, 0, 1)  # (C,H,W)
        
        # Expand to batch dimension
        return pos.unsqueeze(0).repeat(b, 1, 1, 1)


class _DeformEncoderLayer(nn.Module):
    """Deformable encoder block: MSDeformAttn + FFN (with residuals & norms)."""
    def __init__(self, d: int, n_heads: int, n_lvls: int, n_pts: int, ffn: int = 1024, drop: float = 0.1):
        super().__init__()
        # Multi-scale deformable attention for cross-level feature aggregation
        self.self_attn = MSDeformAttn(d_model=d, n_levels=n_lvls, n_heads=n_heads, n_points=n_pts)
        self.norm1 = nn.LayerNorm(d)
        self.drop1 = nn.Dropout(drop)
        
        # Feed-forward network for feature transformation
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.norm2 = nn.LayerNorm(d)
        self.drop2 = nn.Dropout(drop)

    @staticmethod
    def _with_pos(x: torch.Tensor, pos: Optional[torch.Tensor]) -> torch.Tensor:
        """Add positional encoding to input if provided."""
        return x if pos is None else x + pos

    def forward(self, src: torch.Tensor, pos: torch.Tensor, refpts: torch.Tensor,
                spatial_shapes: List[Tuple[int,int]]) -> torch.Tensor:
        """
        Forward pass with deformable attention and FFN.
        
        Args:
            src: Source features (B, Len, D)
            pos: Positional encodings (B, Len, D)
            refpts: Reference points for deformable attention (B, Len, n_levels, 2)
            spatial_shapes: List of (H, W) tuples for each feature level
            
        Returns:
            Transformed features (B, Len, D)
        """
        # Apply multi-scale deformable attention with residual connection
        attn = self.self_attn(self._with_pos(src, pos), refpts, src, spatial_shapes, None)
        src = self.norm1(src + self.drop1(attn))
        
        # Apply feed-forward network with residual connection
        ffn = self.fc2(F.relu(self.fc1(src)))
        src = self.norm2(src + self.drop2(ffn))
        return src


class _PixelDecoderDeformable(nn.Module):
    """
    Pixel decoder: 1x1 unify -> deformable encoder over concatenated multi-level tokens.
    Outputs:
      - mask_features: highest-res map projected to D
      - tokens/pos per level for the decoder memory
    Why: deformable encoder aggregates content across pyramid levels, aligning with Mask2Former's pixel-decoder spirit.
    """
    def __init__(self, in_channels: List[int], d: int = 256, n_heads: int = 8,
                 n_pts: int = 4, n_enc_layers: int = 6, num_out: int = 3):
        super().__init__()
        assert len(in_channels) >= num_out
        self.num_out = num_out
        self.d = d
        
        # 1x1 convolutions to project each level to unified dimension
        self.proj = nn.ModuleList([nn.Conv2d(c, d, 1) for c in in_channels[-num_out:]])
        
        # Learnable embeddings to distinguish different feature levels
        self.level_embed = nn.Parameter(torch.randn(num_out, d))
        
        # Positional encoding generator
        self.pos = _SinePosEnc2D(d)
        
        # Stack of deformable encoder layers for multi-scale feature fusion
        self.enc = nn.ModuleList([_DeformEncoderLayer(d, n_heads, num_out, n_pts) for _ in range(n_enc_layers)])
        
        # Final projection for mask features
        self.mask_proj = nn.Conv2d(d, d, 1)

    @staticmethod
    def _spatial_shapes(tensors: List[torch.Tensor]) -> List[Tuple[int,int]]:
        """Extract spatial dimensions (H, W) from list of feature tensors."""
        return [(t.shape[-2], t.shape[-1]) for t in tensors]

    def _make_ref_points(self, spatial_shapes: List[Tuple[int,int]], B: int, device, dtype) -> torch.Tensor:
        """
        Build reference points for deformable attention.
        
        Args:
            spatial_shapes: List of (H, W) for each level
            B: Batch size
            device: Target device
            dtype: Target dtype
            
        Returns:
            Reference points of shape (B, Len, n_levels, 2) with normalized [0,1] coordinates
        """
        ref_all = []
        for (H, W) in spatial_shapes:
            # Create normalized grid centers for each spatial location
            yy, xx = torch.meshgrid(
                torch.linspace(0.5/H, 1-0.5/H, H, device=device, dtype=dtype),
                torch.linspace(0.5/W, 1-0.5/W, W, device=device, dtype=dtype),
                indexing='ij'
            )
            # Stack x, y coordinates and flatten spatial dimensions
            ref_all.append(torch.stack((xx, yy), -1).reshape(-1, 2))  # (HW,2)
        
        # Concatenate all levels
        ref = torch.cat(ref_all, 0)  # (Len,2)
        
        # Repeat reference points across all levels (for cross-level attention)
        ref = ref[:, None, :].repeat(1, len(spatial_shapes), 1)       # (Len,n_levels,2)
        
        # Expand to batch dimension
        return ref.unsqueeze(0).repeat(B, 1, 1, 1)

    def forward(self, xs: List[torch.Tensor]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """
        Process multi-scale features through deformable pixel decoder.
        
        Args:
            xs: List of feature maps from backbone, ordered low-to-high resolution
            
        Returns:
            Dictionary containing:
                - mask_features: Highest resolution features for mask prediction
                - tokens: Per-level feature tokens
                - pos: Per-level positional encodings
                - shapes: Spatial shapes of each level
        """
        # Project all levels to unified dimension
        feats = [m(x) for m, x in zip(self.proj, xs[-self.num_out:])]  # low->high
        
        # Flatten spatial dimensions and generate positional encodings
        tokens = [f.flatten(2).transpose(1, 2) for f in feats]
        poss   = [self.pos(f).flatten(2).transpose(1, 2) for f in feats]
        spatial_shapes = self._spatial_shapes(feats)
        B, D = feats[0].shape[0], feats[0].shape[1]

        # Concatenate all levels into single sequence (Len = sum of all HW)
        src = torch.cat(tokens, 1)   # (B,Len,D)
        pos = torch.cat(poss, 1)     # (B,Len,D)
        
        # Add learnable level embeddings to distinguish different scales
        start = 0
        for i, (H, W) in enumerate(spatial_shapes):
            L = H * W
            src[:, start:start+L, :] += self.level_embed[i]
            start += L

        # Generate reference points for deformable attention
        refpts = self._make_ref_points(spatial_shapes, B, src.device, src.dtype)  # (B,Len,Lvls,2)
        
        # Apply deformable encoder layers for multi-scale feature aggregation
        for layer in self.enc:
            src = layer(src, pos, refpts, spatial_shapes)

        # Split concatenated features back to per-level representations
        outs = []
        start = 0
        for (H, W) in spatial_shapes:
            L = H * W
            y = src[:, start:start+L, :].transpose(1, 2).reshape(B, D, H, W)
            outs.append(y)
            start += L

        # Use highest resolution level for mask features
        mask_features = self.mask_proj(outs[-1])  # highest-res
        
        # Prepare per-level tokens and positional encodings for decoder
        dec_tokens = [o.flatten(2).transpose(1, 2) for o in outs]
        dec_pos    = [self.pos(o).flatten(2).transpose(1, 2) for o in outs]
        
        return {"mask_features": mask_features, "tokens": dec_tokens, "pos": dec_pos, "shapes": spatial_shapes}
    

class _MaskedXAttn(nn.Module):
    """Masked cross-attn: queries attend only where their masks are confident. Stabilizes instance separation."""
    def __init__(self, d: int, heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        
        # Query, key, value, and output projections
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.drop = dropout

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_soft_mask: torch.Tensor) -> torch.Tensor:
        """
        Masked cross-attention with soft attention mask.
        
        Args:
            q: Query tensor (B, Q, D)
            k: Key tensor (B, K, D)
            v: Value tensor (B, K, D)
            attn_soft_mask: Soft attention mask in [0,1] range (B, Q, K)
            
        Returns:
            Attended features (B, Q, D)
        """
        B, Q, D = q.shape
        K = k.shape[1]
        
        # Project and reshape to multi-head format
        q = self.q(q).view(B, Q, self.h, self.dh).transpose(1, 2)  # (B,H,Q,dh)
        k = self.k(k).view(B, K, self.h, self.dh).transpose(1, 2)  # (B,H,K,dh)
        v = self.v(v).view(B, K, self.h, self.dh).transpose(1, 2)  # (B,H,K,dh)
        
        # Convert soft mask [0,1] to additive mask via log for attention computation
        eps = 1e-6
        additive = torch.log(attn_soft_mask.clamp(min=eps)).unsqueeze(1)  # (B,1,Q,K), broadcast over heads
        
        # Apply scaled dot-product attention with mask
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=additive, dropout_p=self.drop, is_causal=False)
        
        # Reshape back to (B, Q, D) and apply output projection
        out = out.transpose(1, 2).reshape(B, Q, D)
        return self.o(out)


class _DecoderLayer(nn.Module):
    """Self-attn -> masked cross-attn -> FFN."""
    def __init__(self, d: int, heads: int = 8, ffn: int = 1024, dropout: float = 0.1):
        super().__init__()
        # Self-attention for query-to-query interaction
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.n1 = nn.LayerNorm(d)
        self.d1 = nn.Dropout(dropout)
        
        # Masked cross-attention for query-to-memory interaction
        self.xattn = _MaskedXAttn(d, heads, dropout)
        self.n2 = nn.LayerNorm(d)
        self.d2 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.fc1 = nn.Linear(d, ffn)
        self.fc2 = nn.Linear(ffn, d)
        self.n3 = nn.LayerNorm(d)
        self.d3 = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, mem: torch.Tensor, attn_soft_mask: torch.Tensor) -> torch.Tensor:
        """
        Decoder layer forward pass.
        
        Args:
            q: Query embeddings (B, Q, D)
            mem: Memory features from encoder (B, K, D)
            attn_soft_mask: Soft mask for cross-attention (B, Q, K)
            
        Returns:
            Updated query embeddings (B, Q, D)
        """
        # Self-attention among queries with residual connection
        x, _ = self.self_attn(q, q, q)
        q = self.n1(q + self.d1(x))
        
        # Masked cross-attention to memory with residual connection
        x = self.xattn(q, mem, mem, attn_soft_mask)
        q = self.n2(q + self.d2(x))
        
        # Feed-forward network with residual connection
        x = self.fc2(F.relu(self.fc1(q)))
        q = self.n3(q + self.d3(x))
        return q


class Mask2FormerHead(nn.Module):
    """
    Faithful Mask2Former-like instance segmentation head.
    Why these choices:
      • Deformable pixel-encoder aggregates multi-scale features content-aware (robust mask features).
      • Masked cross-attn gates each query to its own spatial support, improving instance separation.
      • Per-layer aux predictions stabilize training (deep supervision).
    """
    def __init__(self, ch: List[int], nc: int, num_queries: int = 100, dim: int = 256,
                 nheads: int = 8, n_dec_layers: int = 6, n_enc_layers: int = 6, n_points: int = 4):
        super().__init__()
        self.nc, self.num_queries = nc, num_queries
        
        # Pixel decoder for multi-scale feature aggregation
        self.pixel = _PixelDecoderDeformable(ch, d=dim, n_heads=nheads, n_pts=n_points, n_enc_layers=n_enc_layers)
        
        # Learnable query embeddings for instance detection
        self.query_embed = nn.Embedding(num_queries, dim)
        
        # Stack of decoder layers
        self.layers = nn.ModuleList([_DecoderLayer(dim, nheads) for _ in range(n_dec_layers)])
        
        # Per-layer classification heads for deep supervision
        self.cls_heads  = nn.ModuleList([nn.Linear(dim, nc) for _ in range(n_dec_layers)])
        
        # Per-layer mask embedding heads for deep supervision
        self.mask_heads = nn.ModuleList([MLP(dim, dim, dim, 3) for _ in range(n_dec_layers)])

    def _build_soft_mask(self, pred_masks: torch.Tensor, shapes: List[Tuple[int,int]]) -> torch.Tensor:
        """
        Build soft attention mask from predicted masks for cross-attention gating.
        
        Args:
            pred_masks: Predicted mask logits (B, Q, Hh, Wh) at highest resolution
            shapes: List of (H, W) tuples for each feature level
            
        Returns:
            Soft attention mask (B, Q, K) where K is concatenated length of all levels
            
        Reason: Memory tokens are concatenated level-wise; this keeps gating aligned per level.
        """
        B, Q, Hh, Wh = pred_masks.shape
        per_level = []
        
        # Resize masks to each level's resolution and flatten
        for (H, W) in shapes:
            m = F.interpolate(pred_masks.sigmoid(), size=(H, W), mode="bilinear", align_corners=False)  # (B,Q,H,W)
            per_level.append(m.flatten(2))  # (B,Q,HW)
        
        # Concatenate all levels to match memory token order
        return torch.cat(per_level, dim=-1)  # (B,Q,K)

    def forward(self, x: List[torch.Tensor]) -> Dict[str, torch.Tensor | List[Dict[str, torch.Tensor]]]:
        """
        Forward pass for instance segmentation.
        
        Args:
            x: List of multi-scale feature maps from backbone
            
        Returns:
            Dictionary containing:
                - pred_logits: Classification logits for each query (B, Q, C)
                - pred_masks: Mask predictions for each query (B, Q, H, W)
                - aux_outputs: List of intermediate predictions for deep supervision
        """
        # Process features through pixel decoder
        feats = self.pixel(x)
        
        # Combine tokens and positional encodings for decoder memory
        mem = torch.cat([t + p for t, p in zip(feats["tokens"], feats["pos"])], dim=1)  # (B,K,D)
        shapes = feats["shapes"]
        B = x[0].shape[0]
        
        # Initialize query embeddings
        q = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B,Q,D)

        aux = []
        
        # Initialize with global attention (all queries attend to all memory)
        attn_soft_mask = torch.ones((B, q.shape[1], mem.shape[1]), device=mem.device, dtype=mem.dtype)
        
        # Iteratively refine queries through decoder layers
        for i, layer in enumerate(self.layers):
            # Update queries through self-attention, cross-attention, and FFN
            q = layer(q, mem, attn_soft_mask)
            
            # Predict class logits for each query
            cls = self.cls_heads[i](q)                # (B,Q,C)
            
            # Generate mask embeddings and compute mask predictions
            m_embed = self.mask_heads[i](q)           # (B,Q,D)
            masks = torch.einsum("bqd,bdhw->bqhw", m_embed, feats["mask_features"])  # (B,Q,Hh,Wh)
            
            # Update attention mask based on predicted masks (except for last layer)
            if i < len(self.layers) - 1:
                with torch.no_grad():
                    attn_soft_mask = self._build_soft_mask(masks, shapes)  # (B,Q,K)
                # Store intermediate predictions for auxiliary loss
                aux.append({"pred_logits": cls, "pred_masks": masks})
        
        # Return final predictions and auxiliary outputs
        return {"pred_logits": cls, "pred_masks": masks, "aux_outputs": aux}
