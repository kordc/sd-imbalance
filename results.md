| Method Name | Train Accuracy | Train Balanced Accuracy | Valid Accuracy | Valid Balanced Accuracy | Test Accuracy | Test Balanced Accuracy | WandB Link |
|-------------|----------------|-------------------------|----------------|-------------------------|---------------|------------------------|------------|
| ResNet18 Clean Training |	0.867 |	0.869 |	0.730 |	0.729 |	0.721 |	0.720 |
|             |                |                         |                |                         |               |                        |            |
|             |                |                         |                |                         |               |                        |            |


1. Pure training config (ResNet18):
- downsample_class: null
- downsample_ratio: 1.0
- epochs: 10
- batch_size: 256
- learning_rate: 0.01
- val_size: 0.2
- num_workers: 2
- augmentations:
  - name: ToTensor
  - name: Normalize
    params:
      mean: [0.5, 0.5, 0.5]
      std: [0.5, 0.5, 0.5]