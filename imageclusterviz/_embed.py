from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.models as tvm
from typing import Callable, Literal
from pathlib import Path


_MODEL_REGISTRY: dict[str, dict] = {
    "resnet50": {
        "ctor": lambda _: tvm.resnet50(weights=None),
        "default": tvm.ResNet50_Weights.IMAGENET1K_V2,
    },
    "vit_b16": {
        "ctor": lambda _: tvm.vit_b_16(weights=None),
        "default": tvm.ViT_B_16_Weights.IMAGENET1K_V1,
    },
    "mobilenetv3": {
        "ctor": lambda _: tvm.mobilenet_v3_large(weights=None),
        "default": tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    },
    # Add new model below
}


# Register model decorator left.


def _pil_preprocess(side: int = 224) -> T.Compose:

    return T.Compose(
        [
            T.Resize(side, antialias=True),
            T.CenterCrop(side),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_model(
    name: str = "resnet50",
    weights: str | None = None,
    checkpoint: str | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
) -> torch.nn.Module:
    """
    Returns a frozen backbone on device.
    """

    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name}."
            f"Available models are: {','.join(_MODEL_REGISTRY)}"
        )

    entry = _MODEL_REGISTRY[name]
    model = entry["ctor"](weights)

    # Load offical weights or custom checkpoints
    if checkpoint is not None:
        state = torch.load(Path(checkpoint), map_location="cpu")
        model.load_state_dict(state, strict=False)
    else:
        if weights is None:
            weights = entry["default"]
        if weights is not None:
            model = entry["ctor"](weights)

    return model.eval().to(device)


def _embed_tensor(model: torch.nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(tensor).flatten(1)


def embed_image(
    path: str | Path,
    *,
    model: str = "resnet50",
    weights: str | None = None,
    checkpoint: str | Path | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
    preprocess: Callable[[Image.Image], torch.Tensor] | None = None,
) -> np.ndarray:
    """Embeds one image"""

    if preprocess is None:
        preprocess = _pil_preprocess()

    net = load_model(model, weights=weights, checkpoint=checkpoint, device=device)
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)

    return _embed_tensor(net, img).cpu().numpy()[0]


def embed_dir(
    img_dir: str | Path,
    *,
    model: str = "resnet50",
    weights: str | None = None,
    checkpoint: str | Path | None = None,
    device: Literal["cpu", "cuda"] = "cpu",
) -> np.ndarray:
    """Embeds all images in image dir"""
    net = load_model(model, weights=weights, checkpoint=checkpoint, device=device)
    preprocess = _pil_preprocess()
    vecs, paths = [], []
    for p in Path(img_dir).iterdir():
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
        vecs.append(_embed_tensor(net, img).cpu())
        paths.append(p.name)

    return np.vstack(vecs), paths
