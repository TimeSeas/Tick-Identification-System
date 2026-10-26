"""Retain v3 proposals, verify their transformed ROIs with a fine-tuned model."""
from collections import OrderedDict

import torch

from app.services.detector import Detector, _dedup


class VerifiedDetector:
    CONFIDENCE = .5

    def __init__(self, proposal_path, verifier_path, tile=2):
        self.tile = tile
        self.proposals = Detector(proposal_path, tile=tile)
        self.proposals.model.roi_heads.mask_roi_pool = None
        verifier = Detector(verifier_path, tile=tile)
        self.model = verifier.model
        self.model.roi_heads.mask_roi_pool = None
        self.device = verifier.device

    @torch.inference_mode()
    def detect(self, img_rgb, tile=None):
        candidates = self.proposals.detect(img_rgb, tile=self.tile if tile is None else tile)
        if not candidates:
            return []
        image = torch.as_tensor(img_rgb.copy(),device=self.device).permute(2,0,1).float()/255
        boxes = torch.tensor([d['bbox'] for d in candidates],dtype=torch.float32,device=self.device)
        images, targets = self.model.transform([image], [dict(boxes=boxes)])
        features = self.model.backbone(images.tensors)
        if isinstance(features,torch.Tensor):
            features = OrderedDict([('0',features)])
        heads = self.model.roi_heads
        pooled = heads.box_roi_pool(features,[targets[0]['boxes']],images.image_sizes)
        logits, _ = heads.box_predictor(heads.box_head(pooled))
        probabilities = logits.softmax(-1)[:,1].cpu().tolist()
        return _dedup([dict(bbox=d['bbox'],score=round(p,4))
                       for d,p in zip(candidates,probabilities) if p>=self.CONFIDENCE])
