import os
import json
from typing import Any, Callable, Optional, Tuple, Union, List
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def default_loader(path: str) -> Any:
    return pil_loader(path)


class ImageRegressionFolder(Dataset):
    """
    A dataset for regression tasks using images and JSON annotations.

    Args:
        root (str or Path): Root directory path containing images.
        annotations_file (str or Path): Path to a JSON file with filename-to-target mappings.
        transform (callable, optional): Transform applied to input images (e.g., ClassificationPresetTrain).
        target_transform (callable, optional): Transform applied to targets.
        loader (callable, optional): Function to load an image from a path.
    """

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        annotations_file: Optional[Union[str, Path]] = None,
    ):
        self.root = root
        self.annotations = self._load_annotations(annotations_file) if annotations_file else {}
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        # Prepare the list of (image_path, target) tuples
        self.samples = self._make_dataset()

    def _load_annotations(self, annotations_file: Union[str, Path]) -> dict:
        with open(annotations_file, "r") as f:
            return json.load(f)

    def _make_dataset(self) -> List[Tuple[str, torch.Tensor]]:
        samples = []
        for filename, targets in self.annotations.items():
            path = os.path.join(self.root, filename)
            if os.path.isfile(path) and self._is_valid_file(path):
                if "loaded" not in targets.keys() or "confidence" not in targets.keys():
                    print("TARGET: ", targets, path)
                target_tensor = torch.tensor(
                    [targets["loaded"], targets["confidence"]],
                    dtype=torch.float32
                )
                samples.append((path, target_tensor))
        return samples

    def _is_valid_file(self, path: str) -> bool:
        return path.lower().endswith(IMG_EXTENSIONS)

    def __getitem__(self, index: int) -> Tuple[Any, torch.Tensor]:
        path, target = self.samples[index]
        image = self.loader(path)

        # Apply image transformations (e.g., ClassificationPresetTrain)
        if self.transform is not None:
            image = self.transform(image)

        # Apply any target transformations
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.samples)
