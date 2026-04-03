from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelConfig:
    backbone: str = "vit_base_patch16_224"
    aggregator: str = "attention"
    hidden_dim: int = 256
    dropout: float = 0.3
    pretrained: bool = True
    freeze_backbone: bool = True
    
    use_dino_segmentation: bool = False
    dino_segmentation_model: str = "vit_small_patch16_224.dino"
    dino_segmenation_threshold: float = 0.6
    
    use_tabular: bool = False
    tabular_features: tuple = ("cirmumference_cm", )
    tabular_hidden_dim: int = 64
    
@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 4
    lr: float = 5e-5
    weight_decay: float = 0.0001
    patience: int = 10
    cv_folds: int = 3
    resume: bool = True
    
    
@dataclass
class DataConfig:
    image_size: int = 224
    num_workers: int = 2
    image_extensions: tuple = (".jpg", ".jpeg", ".png")
    csv_encoding: str = "cp1250"
    csv_sep: str = ";"
    augmentation: str = "light" 

@dataclass
class PathConfig:
    data_dir: Path = Path("data")
    image_dir: Path = Path("data")
    csv_path: Path = Path("data/birch_trees_bratislava.csv")
    output_dir: Path = Path("outputs")
    checkpoint_dir: Path = None
    log_dir: Path = None

@dataclass
class Config:
    model: ModelConfig = None
    training: TrainConfig = None
    data: DataConfig = None
    paths: PathConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.paths is None:
            self.paths = PathConfig()
        
        model_tag = f"{self.model.backbone}_{self.model.aggregator}"
        if self.model.use_dino_segmentation:
            model_tag += "_dino_segmentation"
        if self.model.use_tabular:
            model_tag += "_tabular"
        
        if self.paths.checkpoint_dir is None:
            self.paths.checkpoint_dir = Path(f"outputs/{model_tag}/checkpoints")
        if self.paths.log_dir is None:
            self.paths.log_dir = Path(f"outputs/{model_tag}/logs")
        

        self.paths.output_dir.mkdir(parents=True,exist_ok=True)
        self.paths.checkpoint_dir.mkdir(parents=True,exist_ok=True)
        self.paths.log_dir.mkdir(parents=True,exist_ok=True)
        
    def display(self):
        print("="*40)
        print("Configuration")
        print("="*40)
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
        print("="*40)
        print(f"epochs: {self.training.epochs}")
        print(f"batch_size: {self.training.batch_size}")
        print(f"lr: {self.training.lr}")
        print(f"weight_decay: {self.training.weight_decay}")
        print(f"patience: {self.training.patience}")
        print(f"cv_folds: {self.training.cv_folds}")
        print("="*40)
        print(f"image_size: {self.data.image_size}")
        print(f"image_dir: {self.paths.image_dir}")
        print(f"csv_path: {self.paths.csv_path}")
        print(f"checkpoint_dir: {self.paths.checkpoint_dir}")
        print(f"log_dir: {self.paths.log_dir}")
        
        print("="*40)

class EfficientNetAttention(Config):
    def __post_init__(self):
        self.model = ModelConfig(
            backbone="efficientnet_b2",
            aggregator="attention"
        )
        self.training = TrainConfig()
        self.data = DataConfig()
        self.paths = PathConfig()
        super().__post_init__()
        
class ResNet50Mean(Config):
    def __post_init__(self):
        self.model = ModelConfig(
            backbone="resnet50",
            aggregator="mean"
        )
        self.training = TrainConfig(lr=5e-5)
        self.data = DataConfig()
        self.paths = PathConfig()
        super().__post_init__()
        
class ViTAttention(Config):
    def __post_init__(self):
        self.model = ModelConfig(
            backbone="vit_small_patch16_224",
            aggregator="attention",
            freeze_backbone=True
        )
        self.training = TrainConfig(lr=5e-5, epochs = 60)
        self.data = DataConfig()
        self.paths = PathConfig()
        super().__post_init__()

class DINOBackbone(Config):
    def __post_init__(self):
        self.model = ModelConfig(
            backbone="vit_small_patch16_224.dino",
            aggregator="attention",
            use_dino_segmentation=True,
            dino_segmentation_model="vit_small_patch16_224.dino",
            dino_segmenation_threshold=0.6
        )
        self.training = TrainConfig()
        self.data = DataConfig()
        self.paths = PathConfig()
        super().__post_init__()

class WithTabular(Config):
    def __post_init__(self):
        self.model = ModelConfig(
            backbone="vit_small_patch16_224",
            aggregator="attention",
            use_tabular=True,
            tabular_features=("cirmumference_cm",),
            tabular_hidden_dim=64
        )
        self.training = TrainConfig()
        self.data = DataConfig()
        self.paths = PathConfig()
        super().__post_init__()

class DINOWithTabular(Config):
    def __post_init__(self):
        self.model = ModelConfig(
            backbone="vit_small_patch16_224.dino",
            aggregator="attention",
            use_dino_segmentation=True,
            use_tabular=True,
            tabular_features=("cirmumference_cm", "fungal_infection"),
            tabular_hidden_dim=64
        )
        self.training = TrainConfig()
        self.data = DataConfig()
        self.paths = PathConfig()
        super().__post_init__()

if __name__ == "__main__":
    cfg = Config()
    cfg.display()