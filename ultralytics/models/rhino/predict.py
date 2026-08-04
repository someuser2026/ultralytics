# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics.engine.results import Results
from ultralytics.models.rtdetr.predict import RTDETROBBPredictor
from ultralytics.utils import ops

from .postprocess import rhino_postprocess


class RHINOOBBPredictor(RTDETROBBPredictor):
    """RHINO OBB predictor with reference flattened multiclass top-k output."""

    @staticmethod
    def _head(model):
        """Resolve the RHINO head through predictor/backend wrappers."""
        module = model
        for _ in range(2):
            candidate = getattr(module, "model", None)
            if candidate is None:
                break
            module = candidate
        return module[-1] if hasattr(module, "__getitem__") else None

    def postprocess(self, preds, img, orig_imgs):
        """Return NMS-free RHINO OBB results, permitting multiple labels per query."""
        predictions = preds[0] if isinstance(preds, (list, tuple)) else preds
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        image_shapes = [image.shape[:2] for image in orig_imgs]
        head = self._head(self.model)
        processed = rhino_postprocess(
            predictions,
            image_shapes,
            conf=self.args.conf,
            classes=self.args.classes,
            max_candidates=int(getattr(head, "max_candidates", 500)),
        )
        results = []
        for prediction, original, path in zip(processed, orig_imgs, self.batch[0]):
            obb = torch.cat(
                (
                    prediction["bboxes"],
                    prediction["conf"][:, None],
                    prediction["cls"].to(prediction["bboxes"].dtype)[:, None],
                ),
                dim=-1,
            )
            results.append(Results(original, path=path, names=self.model.names, obb=obb))
        return results
