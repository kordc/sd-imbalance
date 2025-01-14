| Method Name         | Test Accuracy | Test Balanced Accuracy | Airplane | Automobile | Bird | Cat | Deer | Dog | Frog | Horse | Ship | Truck |
|---------------------|---------------|------------------------|----------|------------|------|-----|------|-----|------|-------|------|-------|
| Full data           | 0.855         | 0.853                  | 0.870    | 0.921      | 0.820 | 0.707 | 0.865 | 0.778 | 0.897 | 0.868 | 0.924 | 0.904 |
| Cat 10% Downsample  | 0.896         | 0.837                  | TBD      | TBD        | TBD  | TBD | TBD  | TBD | TBD  | TBD   | TBD  | TBD   |
| Cat 1% Downsample   | 0.807         | 0.807                  | 0.890    | 0.926      | 0.835 | 0.000 | 0.874 | 0.879 | 0.928 | 0.899 | 0.928 | 0.915 |




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
2. Cat 1% Downsample:
    downsample_class: cat # Class to downsample, e.g., "cat"
    downsample_ratio: 0.01 # How many examples to keep\
3. Cat 10% Downsample"
    downsample_class: cat # Class to downsample, e.g., "cat"
    downsample_ratio: 0.1 # How many examples to keep\
