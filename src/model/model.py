import torch 
import torch.nn as nn
import torch.nn.functional as F
import timm
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.config import Config

def get_backbone(name, pretrained=True):
    backbone = timm.create_model(
        name,
        pretrained=pretrained,
        num_classes=0,  # Remove the classification head
        global_pool="avg"  # Use global average pooling
    )
    
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 224, 224)
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
        freeze_backbone=False
    ):
        super().__init__()
        self.backbone, feature_dim = get_backbone(backbone_name, pretrained)
        self.aggregator = get_aggregator(aggregator_name, feature_dim)
        self.regression_head = RegressionHead(feature_dim, hidden_dim, dropout)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen. Only aggregator and regression head will be trained.")
            
    def forward_single(self, images):
        features = self.backbone(images.to(next(self.parameters()).device))  # (N, feature_dim)
        aggregated = self.aggregator(features)  # (feature_dim,)
        prediction = self.regression_head(aggregated)  # (1,)
        return prediction
    
    def forward(self, images):
        predictions = [self.forward_single(imgs) for imgs in images]
        return torch.stack(predictions)
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen. All parameters will be trained.")
        
    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}
    
def build_model(cfg: Config):
    return BirchVitalityModel(
        backbone_name=cfg.model.get("backbone", "efficientnet_b0"),
        aggregator_name=cfg.model.get("aggregator", "attention"),
        hidden_dim=cfg.model.get("hidden_dim", 256),
        dropout=cfg.model.get("dropout", 0.3),
        pretrained=cfg.model.get("pretrained", True),
        freeze_backbone=cfg.model.get("freeze_backbone", False)
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