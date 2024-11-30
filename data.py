import numpy as np
import torchvision


class DownsampledCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, download=False, downsample_class=None, downsample_ratio=1.0):
        super().__init__(root=root, train=train, transform=transform, download=download)
        self.downsample_class = downsample_class
        self.downsample_ratio = downsample_ratio
        if downsample_class is not None:
            self._downsample()

    def _downsample(self):
        targets = np.array(self.targets)
        selected_idx = np.arange(len(targets))

        # Check if the specified class exists in the dataset
        if self.downsample_class in targets:
            # Get indices of all samples of the target class
            class_idx = selected_idx[targets == self.downsample_class]
            # Number of samples to keep for the target class
            keep_size = int(len(class_idx) * self.downsample_ratio)
            # Randomly sample indices to keep
            keep_idx = np.random.choice(
                class_idx, size=keep_size, replace=False)
            # Combine with indices of all other classes
            non_class_idx = selected_idx[targets != self.downsample_class]
            final_idx = np.concatenate([non_class_idx, keep_idx])
            # Update the dataset
            self.data = self.data[final_idx]
            self.targets = list(targets[final_idx])
        else:
            print(
                f"Class {self.downsample_class} not found in the dataset. Skipping downsampling.")
