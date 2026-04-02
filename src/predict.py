import sys
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import transforms

sys.path.append(str(Path(__file__).parent))

from config.config import Config
from model.model import BirchVitalityModel

def load_images(image_path: Path, image_size: int = 224) -> torch.Tensor:
    """
    Loads all images from a folder or a single image file.
    Returns tensor of shape (N, 3, H, W).
    """
    image_path = Path(image_path)
    extensions = ('.jpg', '.jpeg', '.png')

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    if image_path.is_file():
        paths = [image_path]
    elif image_path.is_dir():
        paths = sorted([
            p for p in image_path.iterdir()
            if p.suffix.lower() in extensions
        ])
    else:
        raise FileNotFoundError(f"Path not found: {image_path}")

    if len(paths) == 0:
        raise ValueError(f"No images found in {image_path}")

    images = []
    for p in paths:
        try:
            img = Image.open(p).convert('RGB')
            img = ImageOps.exif_transpose(img)
            img = transform(img)
            images.append(img)
            print(f" Loaded: {p.name}")
        except Exception as e:
            print(f" Warning: could not load {p.name}: {e}")

    if len(images) == 0:
        raise ValueError("No images could be loaded successfully.")

    return torch.stack(images)


def predict(image_path: str, checkpoint_path: str = None, config: Config = None):
    """
    Predict vitality score for a single tree.

    Args:
        image_path: path to folder of images or single image file
        checkpoint_path: path to model weights (default: best_model.pt)
        config: Config instance (default: Config())

    Returns:
        float vitality prediction (1.0–5.0)
    """
    if config is None:
        config = Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if checkpoint_path is None:
        checkpoint_path = config.paths.checkpoint_dir / "best_model.pt"
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run main.py first to train and save the model."
        )

    model = BirchVitalityModel(
        backbone_name = config.model.backbone,
        aggregator_name = config.model.aggregator,
        hidden_dim = config.model.hidden_dim,
        dropout = config.model.dropout,
        pretrained = False,
    )
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"Backbone: {config.model.backbone} | Aggregator: {config.model.aggregator}\n")

    print(f"Loading images from: {image_path}")
    images = load_images(image_path, config.data.image_size)
    print(f"Loaded {len(images)} image(s)\n")

    images = images.to(device)
    with torch.no_grad():
        prediction = model.forward_single(images)

    vitality = prediction.item()
    vitality_clipped = max(1.0, min(5.0, vitality))

    print("=" * 40)
    print("PREDICTION")
    print("=" * 40)
    print(f"Raw prediction: {vitality:.4f}")
    print(f"Clipped (1.0–5.0): {vitality_clipped:.2f}")
    print(f"Rounded: {round(vitality_clipped * 2) / 2:.1f} (nearest 0.5 step)")
    print("=" * 40)
    print(f"\nVitality scale:")
    print(f" 1.0 = very poor 2.0 = poor 3.0 = moderate")
    print(f" 4.0 = good 5.0 = excellent")

    return vitality_clipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict birch tree vitality from images")
    parser.add_argument(
        "--images",
        type = str,
        required = True,
        help = "Path to folder of tree images or a single image file"
    )
    parser.add_argument(
        "--checkpoint",
        type = str,
        default = None,
        help = "Path to model checkpoint (default: outputs/<model>/checkpoints/best_model.pt)"
    )
    args = parser.parse_args()

    config = Config()
    predict(
        image_path = args.images,
        checkpoint_path = args.checkpoint,
        config = config,
    )