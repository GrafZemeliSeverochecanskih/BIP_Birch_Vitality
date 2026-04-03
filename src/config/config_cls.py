"""
Separate configuration class for the classification pipeline.
Keeps the regression Config untouched.
"""
from dataclasses import dataclass, field
from pathlib import Path
from config.config import ModelConfig, TrainConfig, DataConfig, PathConfig


@dataclass
class ClassificationConfig:
    model: ModelConfig = None
    training: TrainConfig = None
    data: DataConfig = None
    paths: PathConfig = None
    num_classes: int = 5

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.paths is None:
            self.paths = PathConfig()

        # Build a distinct output tag so classification results
        # never collide with regression results.
        model_tag = f"{self.model.backbone}_{self.model.aggregator}_cls"
        if self.model.use_dino_segmentation:
            model_tag += "_dino_segmentation"
        if self.model.use_tabular:
            model_tag += "_tabular"

        if self.paths.checkpoint_dir is None:
            self.paths.checkpoint_dir = Path(f"outputs/{model_tag}/checkpoints")
        if self.paths.log_dir is None:
            self.paths.log_dir = Path(f"outputs/{model_tag}/logs")

        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)

    def display(self):
        print("=" * 40)
        print("Classification Configuration")
        print("=" * 40)
        print(f"num_classes: {self.num_classes}")
        print(f"backbone: {self.model.backbone}")
        print(f"aggregator: {self.model.aggregator}")
        print(f"hidden_dim: {self.model.hidden_dim}")
        print(f"dropout: {self.model.dropout}")
        print(f"freeze_backbone: {self.model.freeze_backbone}")
        print(f"use_dino_segmentation: {self.model.use_dino_segmentation}")
        if self.model.use_dino_segmentation:
            print(f"dino_segmentation_model: {self.model.dino_segmentation_model}")
            print(f"dino_segmenation_threshold: {self.model.dino_segmenation_threshold}")
        print(f"use_tabular: {self.model.use_tabular}")
        if self.model.use_tabular:
            print(f"tabular_features: {self.model.tabular_features}")
            print(f"tabular_hidden_dim: {self.model.tabular_hidden_dim}")
        print("=" * 40)
        print(f"epochs: {self.training.epochs}")
        print(f"batch_size: {self.training.batch_size}")
        print(f"lr: {self.training.lr}")
        print(f"weight_decay: {self.training.weight_decay}")
        print(f"patience: {self.training.patience}")
        print(f"cv_folds: {self.training.cv_folds}")
        print("=" * 40)
        print(f"image_size: {self.data.image_size}")
        print(f"image_dir: {self.paths.image_dir}")
        print(f"csv_path: {self.paths.csv_path}")
        print(f"checkpoint_dir: {self.paths.checkpoint_dir}")
        print(f"log_dir: {self.paths.log_dir}")
        print("=" * 40)
