import torch 
import torch.nn as nn
import torch.nn.functional as F
import timm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import Config

def _interpolate_pos_embed(pos_embed_ckpt: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Bicubic-interpolate position embeddings from checkpoint resolution to model resolution.

    DINOv2 checkpoints are pretrained at 518×518 (patch 14 → 37×37 = 1369 patches).
    When the model is built with img_size=224 (16×16 = 256 patches) the shapes don't
    match.  This function interpolates the spatial patch tokens and leaves the CLS token
    untouched, exactly mirroring what timm does internally when pretrained=True.
    """
    N_ckpt  = pos_embed_ckpt.shape[1] - 1  # patches in checkpoint (exclude CLS)
    N_model = model.pos_embed.shape[1] - 1  # patches expected by model

    if N_ckpt == N_model:
        return pos_embed_ckpt  # no-op when sizes already match

    cls_token   = pos_embed_ckpt[:, :1, :]  # [1, 1, D]
    patch_embed = pos_embed_ckpt[:, 1:, :]  # [1, N_ckpt, D]

    D     = patch_embed.shape[-1]
    H_old = W_old = int(N_ckpt  ** 0.5)
    H_new = W_new = int(N_model ** 0.5)

    patch_embed = patch_embed.reshape(1, H_old, W_old, D).permute(0, 3, 1, 2)  # [1,D,H,W]
    patch_embed = F.interpolate(patch_embed, size=(H_new, W_new), mode="bicubic", align_corners=False)
    patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, H_new * W_new, D)  # [1,N_new,D]

    print(f"  [pos_embed] interpolated {H_old}×{W_old} → {H_new}×{W_new} patches")
    return torch.cat([cls_token, patch_embed], dim=1)


def _load_dino_weights(model: nn.Module) -> None:
    """
    Download and apply pretrained DINO/DINOv2 weights to a timm ViT created with
    pretrained=False.  Handles two issues that break plain load_state_dict:
      1. norm.weight/bias → fc_norm.weight/bias key rename
      2. pos_embed shape mismatch when img_size differs from pretraining resolution
    """
    pretrained_cfg = model.pretrained_cfg
    if "hf_hub_id" in pretrained_cfg:
        state_dict = timm.models.load_state_dict_from_hf(pretrained_cfg["hf_hub_id"])
    else:
        state_dict = timm.models.load_state_dict_from_url(pretrained_cfg["url"])

    remapped = {}
    for k, v in state_dict.items():
        if k == "norm.weight":
            k = "fc_norm.weight"
        elif k == "norm.bias":
            k = "fc_norm.bias"
        remapped[k] = v

    if "pos_embed" in remapped:
        remapped["pos_embed"] = _interpolate_pos_embed(remapped["pos_embed"], model)

    missing, unexpected = model.load_state_dict(remapped, strict=False)
    if missing:
        print(f"  [DINO load] missing keys (ok if head-related): {missing}")
    if unexpected:
        print(f"  [DINO load] unexpected keys (ok if head-related): {unexpected}")


def get_backbone(name, pretrained=True, img_size=224):
    is_dino = ".dino" in name or "dinov2" in name

    if is_dino and pretrained:
        backbone = timm.create_model(
            name,
            pretrained=False,
            num_classes=0,
            global_pool="avg",
            img_size=img_size,
        )
        _load_dino_weights(backbone)
    else:
        backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            img_size=img_size,
        )

    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size)
        feature_dim = backbone(dummy).shape[-1]

    print(f"Backbone: {name}")
    print(f"Feature dim: {feature_dim}")
    return backbone, feature_dim

class MeanAggregator(nn.Module):
    """
    Simple mean pooling over all image features in the bag.
    Fast, no learnable parameters.
    Input:  (N, feature_dim)
    Output: (feature_dim,)
    """
    def forward(self, x: torch.Tensor):
        return x.mean(dim=0)

class MaxAggregator(nn.Module):
    """
    Max pooling over all image features in the bag.
    Focuses on the most discriminative image.
    Input:  (N, feature_dim)
    Output: (feature_dim,)
    """
    def forward(self, x: torch.Tensor):
        return x.max(dim=0).values
    
class AttentionAggregator(nn.Module):
    """
    Learnable attention-based MIL aggregation.
    Learns which images are most informative for vitality.

    Attention mechanism:
        score_i = softmax( tanh(W * h_i) )
        output  = sum( score_i * h_i )

    Input:  (N, feature_dim)
    Output: (feature_dim,)
    """
    def __init__(self, feature_dim, hidden_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor):
        scores = self.attention(x)  # (N, 1)
        weights = F.softmax(scores, dim=0)  # (N, 1)
        output = (weights * x).sum(dim=0)  # (feature_dim,)
        return output

def get_aggregator(name, feature_dim):
    aggreagators = {
        "mean": MeanAggregator(),
        "max": MaxAggregator(),
        "attention": AttentionAggregator(feature_dim)
    }
    if name not in aggreagators:
        raise ValueError(f"Unknown aggregator: {name}, Choose from {list(aggreagators.keys())}")
    
    print(f"Using aggregator: {name}")
    return aggreagators[name]

class DINOSegmenter(nn.Module):
    def __init__(
        self,
        model_name="vit_small_patch16_224.dino",
        threshold=0.6,
        img_size=224,
    ):
        super().__init__()
        is_dino = ".dino" in model_name or "dinov2" in model_name
        self.vit = timm.create_model(
            model_name,
            pretrained=False if is_dino else True,
            num_classes=0,
            img_size=img_size,
        )
        if is_dino:
            _load_dino_weights(self.vit)
        for p in self.vit.parameters():
            p.requires_grad = False
        self.vit.eval()
        
        ps = self.vit.patch_embed.patch_size
        self.patch_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
        self.threshold = threshold
        self._attn: torch.Tensor | None = None
        
        last_block = self.vit.blocks[-1]
        last_block.attn.fused_attn = False
        
        def _capture_attn(module, input, output):
            self._attn = input[0].detach()
        
        last_block.attn.attn_drop.register_forward_hook(_capture_attn)
        print(f"DINO segmenter: {model_name} | threshold {threshold} | patch size {self.patch_size}")
    
    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        B, C, H, W = images.shape
        h_p = H // self.patch_size
        w_p = W // self.patch_size
        
        self.vit(images)
        
        attn = self._attn                        # (B, num_heads, num_tokens, num_tokens)
        cls_attn = attn[:, :, 0, 1:]                # (B, num_heads, num_patches) — CLS → all patches
        cls_attn = cls_attn.mean(dim=1)             # (B, num_patches) — average over heads
        cls_attn = cls_attn.reshape(B, h_p, w_p)    # (B, h_p, w_p)
        
        a_min = cls_attn.flatten(1).min(dim=1)[0].view(B, 1, 1)
        a_max = cls_attn.flatten(1).max(dim=1)[0].view(B, 1, 1)
        
        cls_attn = (cls_attn - a_min) / (a_max - a_min + 1e-8)
        mask = (cls_attn > self.threshold).float().unsqueeze(1)
        mask = F.interpolate(mask, size=(H, W), mode="nearest")
        
        return images * mask

        
class TabularEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        print(f"TabularEncoder: {n_features} -> {hidden_dim} -> {out_dim}")

    def forward(self, x):
        return self.net(x)


class RegressionHead(nn.Module):
    def __init__(self, feature_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor):
        return self.head(x).squeeze(-1)

class BirchVitalityModel(nn.Module):
    """
    Full MIL model for birch tree vitality regression.

    Forward pass (single tree):
        images (N, 3, H, W)
            -> backbone -> (N, feature_dim)
            -> aggregator -> (feature_dim,)
            -> regression head -> scalar

    Forward pass (batch):
        images: list of tensors, each (N_i, 3, H, W)
            -> process each tree independently
            -> stack predictions -> (B,)

    Args:
        backbone_name:   timm model name (default: "efficientnet_b0")
        aggregator_name: "mean", "max", or "attention"
        hidden_dim:      regression head hidden size
        dropout:         dropout rate in regression head
        pretrained:      use ImageNet pretrained weights
        freeze_backbone: freeze backbone weights (train head only)
    """
    def __init__(
        self,
        backbone_name="efficientnet_b0",
        aggregator_name="attention",
        hidden_dim=256,
        dropout=0.3,
        pretrained=True,
        freeze_backbone=False,
        img_size=224,

        use_dino = False,
        dino_seg_model: str = "vit_small_patch16_224.dino",
        dino_seg_threshold: float = 0.6,

        use_tabular: bool = False,
        n_tabular_features: int = 0,
        tabular_hidden_dim: int = 64
    ):
        super().__init__()

        self.segmenter = (
            DINOSegmenter(dino_seg_model, dino_seg_threshold, img_size=img_size) if use_dino else None
        )

        self.backbone, feature_dim = get_backbone(backbone_name, pretrained, img_size=img_size)
        self.aggregator = get_aggregator(aggregator_name, feature_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen. Only aggregator and regression head will be trained.")
        
        tabular_out_dim = 0
        self.tabular_encoder = None
        if use_tabular and n_tabular_features > 0:
            tabular_out_dim = tabular_hidden_dim
            self.tabular_encoder = TabularEncoder(
                n_tabular_features, tabular_hidden_dim, tabular_out_dim
            )

        self.regression_head = RegressionHead(
            feature_dim + tabular_out_dim, hidden_dim, dropout
        )

    def forward_single(
        self, 
        images,
        tabular
        ):
        if self.segmenter is not None:
            images = self.segmenter(images)

        features = self.backbone(images.to(next(self.parameters()).device))  # (N, feature_dim)
        aggregated = self.aggregator(features)  # (feature_dim,)

        if self.tabular_encoder is not None and tabular is not None:
            tab_emb = self.tabular_encoder(tabular)
            aggregated = torch.cat([aggregated, tab_emb], dim = -1)

        prediction = self.regression_head(aggregated)  # (1,)
        return prediction
    
    def forward(
        self, 
        images,
        tabular
        ):
        if tabular is not None:
            predictions = [
                self.forward_single(imgs, tab)
                for imgs, tab in zip(images, tabular)
            ]
        else:
            predictions = [self.forward_single(imgs, None) for imgs in images]
        return torch.stack(predictions)
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen. All parameters will be trained.")
        
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
    
def build_model(cfg: Config, n_tabular_features: int):
    return BirchVitalityModel(
        backbone_name=cfg.model.get("backbone", "efficientnet_b0"),
        aggregator_name=cfg.model.get("aggregator", "attention"),
        hidden_dim=cfg.model.get("hidden_dim", 256),
        dropout=cfg.model.get("dropout", 0.3),
        pretrained=cfg.model.get("pretrained", True),
        freeze_backbone=cfg.model.get("freeze_backbone", False),
        use_dino_seg = cfg.model.get("use_dino_seg", False),
        dino_seg_model = cfg.model.get("dino_segmentation_model", "vit_small_patch16_224.dino"),
        dino_seg_threshold = cfg.model.get("dino_segmenation_threshold", 0.6),
        use_tabular = cfg.model.get("use_tabular", False),
        n_tabular_features = n_tabular_features,
        tabular_hidden_dim = cfg.model.get("tabular_hidden_dim", 64)
    )

class ClassificationHead(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x: torch.Tensor):
        return self.head(x)  # (num_classes,) — raw logits


class BirchVitalityClassifier(nn.Module):
    """
    MIL classification model for birch tree vitality.

    Identical architecture to BirchVitalityModel (backbone -> aggregator -> head)
    but outputs class logits instead of a scalar regression value.

    Args:
        num_classes:     number of vitality classes (default 5: classes 1–5)
        backbone_name:   timm model name
        aggregator_name: "mean", "max", or "attention"
        hidden_dim:      classification head hidden size
        dropout:         dropout rate
        pretrained:      use ImageNet pretrained weights
        freeze_backbone: freeze backbone weights
        use_dino:        enable DINO attention-based segmentation
        use_tabular:     enable tabular feature fusion
    """
    def __init__(
        self,
        num_classes=5,
        backbone_name="efficientnet_b0",
        aggregator_name="attention",
        hidden_dim=256,
        dropout=0.3,
        pretrained=True,
        freeze_backbone=False,
        img_size=224,

        use_dino=False,
        dino_seg_model: str = "vit_small_patch16_224.dino",
        dino_seg_threshold: float = 0.6,

        use_tabular: bool = False,
        n_tabular_features: int = 0,
        tabular_hidden_dim: int = 64,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.segmenter = (
            DINOSegmenter(dino_seg_model, dino_seg_threshold, img_size=img_size) if use_dino else None
        )

        self.backbone, feature_dim = get_backbone(backbone_name, pretrained, img_size=img_size)
        self.aggregator = get_aggregator(aggregator_name, feature_dim)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen. Only aggregator and classification head will be trained.")

        tabular_out_dim = 0
        self.tabular_encoder = None
        if use_tabular and n_tabular_features > 0:
            tabular_out_dim = tabular_hidden_dim
            self.tabular_encoder = TabularEncoder(
                n_tabular_features, tabular_hidden_dim, tabular_out_dim
            )

        self.classification_head = ClassificationHead(
            feature_dim + tabular_out_dim, num_classes, hidden_dim, dropout
        )

    def forward_single(self, images, tabular):
        if self.segmenter is not None:
            images = self.segmenter(images)

        features = self.backbone(images.to(next(self.parameters()).device))
        aggregated = self.aggregator(features)

        if self.tabular_encoder is not None and tabular is not None:
            tab_emb = self.tabular_encoder(tabular)
            aggregated = torch.cat([aggregated, tab_emb], dim=-1)

        logits = self.classification_head(aggregated)  # (num_classes,)
        return logits

    def forward(self, images, tabular):
        if tabular is not None:
            logits = [
                self.forward_single(imgs, tab)
                for imgs, tab in zip(images, tabular)
            ]
        else:
            logits = [self.forward_single(imgs, None) for imgs in images]
        return torch.stack(logits)  # (B, num_classes)

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen. All parameters will be trained.")

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_classifier(cfg: Config, n_tabular_features: int, num_classes: int = 5):
    return BirchVitalityClassifier(
        num_classes=num_classes,
        backbone_name=cfg.model.get("backbone", "efficientnet_b0"),
        aggregator_name=cfg.model.get("aggregator", "attention"),
        hidden_dim=cfg.model.get("hidden_dim", 256),
        dropout=cfg.model.get("dropout", 0.3),
        pretrained=cfg.model.get("pretrained", True),
        freeze_backbone=cfg.model.get("freeze_backbone", False),
        use_dino_seg=cfg.model.get("use_dino_seg", False),
        dino_seg_model=cfg.model.get("dino_segmentation_model", "vit_small_patch16_224.dino"),
        dino_seg_threshold=cfg.model.get("dino_segmenation_threshold", 0.6),
        use_tabular=cfg.model.get("use_tabular", False),
        n_tabular_features=n_tabular_features,
        tabular_hidden_dim=cfg.model.get("tabular_hidden_dim", 64),
    )


if __name__ == "__main__":
    print("model sanity check:")
    
    model = BirchVitalityModel(
        backbone_name="efficientnet_b0",
        aggregator_name="attention",
        hidden_dim=256,
        dropout=0.3,
        pretrained=False,
    )
    
    mock_batch = [
        torch.randn(3, 3, 224, 224),  # Tree 1 with 3 images
        torch.randn(5, 3, 224, 224),  # Tree 2 with 5 images
        torch.randn(2, 3, 224, 224),  # Tree 3 with 2 images
        torch.randn(7, 3, 224, 224)   # Tree 4 with 7 images
    ]
    
    model.eval()
    with torch.no_grad():
        predictions = model(mock_batch)
    
    print(f"Predictions shape: {predictions.shape}")  # Should be (4,)
    print(f"Output: {predictions}")
    print(f"Shape: {predictions.shape}")
    
    params = model.count_parameters()
    print(f"Total parameters: {params['total']}")
    print(f"Trainable parameters: {params['trainable']}")
    print(f"Frozen parameters: {params['frozen']}")
    
    print("Model sanity check passed!")
    model.unfreeze_backbone()
    model2 = BirchVitalityModel(
        backbone_name="efficientnet_b0",
        aggregator_name="attention",
        hidden_dim=256,
        dropout=0.3,
        pretrained=False,
        freeze_backbone=True
    )
    params2 = model2.count_parameters()
    print(f"Frozen model - Trainable parameters: {params2['trainable']}")
    
    print("Testing all aggregators:")
    for agg in ["mean", "max", "attention"]:
        print(f"\nTesting aggregator: {agg}")
        model = BirchVitalityModel(
            backbone_name="efficientnet_b0",
            aggregator_name=agg,
            pretrained=False,
        )
        with torch.no_grad():
            predictions = model(mock_batch)
        print(f"Predictions shape: {predictions.shape}")  # Should be (4,)
        print(f"Output: {predictions}")