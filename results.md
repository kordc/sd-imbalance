| Method Name         | Test Accuracy | Test Balanced Accuracy | Airplane | Automobile | Bird | Cat | Deer | Dog | Frog | Horse | Ship | Truck |
|---------------------|---------------|------------------------|----------|------------|------|-----|------|-----|------|-------|------|-------|
| Full data           | 0.855         | 0.853                  | 0.870    | 0.921      | 0.820 | 0.707 | 0.865 | 0.778 | 0.897 | 0.868 | 0.924 | 0.904 |
| Cat 10% Downsample  | TBD           | TBD                    | TBD      | TBD        | TBD  | TBD | TBD  | TBD | TBD  | TBD   | TBD  | TBD   |
| Cat 1% Downsample   | TBD           | TBD                    | TBD      | TBD        | TBD  | TBD | TBD  | TBD | TBD  | TBD   | TBD  | TBD   |




1. Pure training config (ResNet18):
    downsample_class: # Class to downsample, e.g., "cat"
    downsample_ratio: 0.01 # How many examples to keep
    name: basic_experiment
    epochs: 100
    batch_size: 128
    learning_rate: 0.1
    val_size: 0.2
    num_workers: 2
    seed: 42
    augmentations:
        - name: ToTensor
        - name: Normalize
          params:
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
        - name: RandomCrop
          params:
              size: 32
              padding: 4
        - name: RandomHorizontalFlip
          params:
              p: 0.5
    test_augmentations:
        - name: ToTensor
        - name: Normalize
          params:
              mean: [0.485, 0.456, 0.406]
              std: [0.229, 0.224, 0.225]
