from torchvision.datasets import MNIST
import os

def prepare_mnist_as_imagefolder(root="./mnist"):
    for split in ["train", "val"]:
        download = split == "train"
        mnist_data = MNIST(root=root, train=(split == "train"), download=download)
        for i, (img, label) in enumerate(mnist_data):
            class_dir = os.path.join(root, split, str(label))
            os.makedirs(class_dir, exist_ok=True)
            img.save(os.path.join(class_dir, f"{i}.png"))

prepare_mnist_as_imagefolder()
