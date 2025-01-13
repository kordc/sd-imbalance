| Method Name        | Train Accuracy | Train Balanced Accuracy | Valid Accuracy | Valid Balanced Accuracy | Test Accuracy | Test Balanced Accuracy | WandB Link |
|--------------------|----------------|-------------------------|----------------|-------------------------|---------------|------------------------|------------|
| Clean Training     | 0.867          | 0.869                  | 0.730          | 0.729                  | 0.721         | 0.720                 |            |
| Cat 10% Downsample | 0.896          | 0.837                  | 0.755          | 0.698                  | 0.689         | 0.689                 |            |
| Cat 1% Downsample  | 0.885          | 0.885                  | 0.739          | 0.739                  | 0.683         | 0.683                 |            |



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

100:
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│       test_accuracy       │    0.8105000257492065     │
│  test_balanced_accuracy   │    0.8105036020278931     │
│         test_loss         │    0.6612717509269714     │
└───────────────────────────┴───────────────────────────┘