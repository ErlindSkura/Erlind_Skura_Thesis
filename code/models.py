"""Model definitions: Mask R-CNN with rescaled anchors, and a U-Net baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator


# --- Mask R-CNN ------------------------------------------------------------

# Anchors are set from the empirical bead size distribution, not left at their
# COCO values. The median bead is 14 px across at 500x and 76 px at 3000x, and
# with the wide random-scaling policy the model sees roughly 7 to 150 px. The
# torchvision default smallest anchor is 32 px, which would leave the smallest
# beads without a well-matched proposal.
ANCHOR_SIZES = (8, 16, 32, 64, 128)

# A single 500x micrograph contains up to 152 beads. The COCO default of 100
# detections per image would silently truncate the count -- an error that would
# corrupt precisely the quantity this work measures.
DETECTIONS_PER_IMG = 400


def build_maskrcnn(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.maskrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone="DEFAULT" if pretrained else None,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_mask, 256, num_classes)

    # One size and three aspect ratios per pyramid level keeps the number of
    # anchors per location at three, which is what the pretrained RPN head
    # already predicts, so the head does not have to be rebuilt.
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=tuple((s,) for s in ANCHOR_SIZES),
        aspect_ratios=((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES),
    )

    model.roi_heads.detections_per_img = DETECTIONS_PER_IMG
    model.rpn._pre_nms_top_n = {"training": 2000, "testing": 2000}
    model.rpn._post_nms_top_n = {"training": 2000, "testing": 1000}
    return model


# --- Faster R-CNN ----------------------------------------------------------


def build_fasterrcnn(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    """Mask R-CNN's detector without the mask branch.

    Included because the laboratory endpoint is a count, and a count needs only
    detection. Comparing the two isolates what the mask branch contributes: the
    two models share a backbone, an anchor configuration and a detection budget,
    so a difference in counting accuracy between them is attributable to the mask
    branch rather than to capacity or to tuning. The same architecture was applied
    to cell counting in low-contrast micrographs by Uka et al. (2020).
    """
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
        weights=weights, weights_backbone="DEFAULT" if pretrained else None,
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model.rpn.anchor_generator = AnchorGenerator(
        sizes=tuple((s,) for s in ANCHOR_SIZES),
        aspect_ratios=((0.5, 1.0, 2.0),) * len(ANCHOR_SIZES),
    )

    model.roi_heads.detections_per_img = DETECTIONS_PER_IMG
    model.rpn._pre_nms_top_n = {"training": 2000, "testing": 2000}
    model.rpn._post_nms_top_n = {"training": 2000, "testing": 1000}
    return model


def set_input_size(model: nn.Module, min_size: int, max_size: int) -> None:
    """Pin the internal resize so that no resampling happens.

    Training crops are square and are passed at native resolution; whole
    micrographs are 1024x736 and are also passed at native resolution. Keeping
    both at scale factor 1.0 means the anchor sizes above mean the same thing at
    training and at test time.
    """
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size


# --- U-Net -----------------------------------------------------------------


class _DecoderBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet34(nn.Module):
    """U-Net with an ImageNet-pretrained ResNet-34 encoder.

    Written out rather than taken from a segmentation library so that the
    baseline has no dependency that could silently change between runs.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        enc = torchvision.models.resnet34(
            weights="DEFAULT" if pretrained else None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)   # 64,  /2
        self.pool = enc.maxpool
        self.layer1, self.layer2 = enc.layer1, enc.layer2         # 64 /4, 128 /8
        self.layer3, self.layer4 = enc.layer3, enc.layer4         # 256 /16, 512 /32

        self.dec4 = _DecoderBlock(512, 256, 256)
        self.dec3 = _DecoderBlock(256, 128, 128)
        self.dec2 = _DecoderBlock(128, 64, 64)
        self.dec1 = _DecoderBlock(64, 64, 32)
        self.dec0 = _DecoderBlock(32, 0, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        s0 = self.stem(x)          # /2
        s1 = self.layer1(self.pool(s0))
        s2 = self.layer2(s1)
        s3 = self.layer3(s2)
        s4 = self.layer4(s3)
        d = self.dec4(s4, s3)
        d = self.dec3(d, s2)
        d = self.dec2(d, s1)
        d = self.dec1(d, s0)
        d = self.dec0(d)
        return self.head(d)


def dice_bce_loss(logits, target, eps: float = 1.0):
    """Cross-entropy plus Dice, to counter the foreground/background imbalance.

    Bead pixels are a small fraction of every micrograph, and cross-entropy
    alone is minimised well by a nearly empty prediction.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    denom = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = 1.0 - ((2 * inter + eps) / (denom + eps)).mean()
    return bce + dice
