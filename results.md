| Method Name                          | Test Accuracy | Test Balanced Accuracy | Airplane | Automobile | Bird  | Cat   | Deer  | Dog   | Frog  | Horse | Ship  | Truck |
|--------------------------------------|---------------|-------------------------|----------|------------|-------|-------|-------|-------|-------|-------|-------|-------|
| Full data                           | 0.855         | 0.853                   | 0.870    | 0.921      | 0.820 | 0.707 | 0.865 | 0.778 | 0.897 | 0.868 | 0.924 | 0.904 |
| Downsampled                         | 0.807         | 0.807                   | 0.890    | 0.926      | 0.835 | 0.000 | 0.874 | 0.879 | 0.928 | 0.899 | 0.928 | 0.915 |
| Downsampled + random-undersample    | 0.217         | 0.217                   | 0.390    | 0.118      | 0.003 | 0.119 | 0.344 | 0.362 | 0.122 | 0.038 | 0.320 | 0.357 |
| Full data + more augmentations      | 0.105         | 0.105                   | 0.904    | 0.000      | 0.022 | 0.006 | 0.004 | 0.016 | 0.003 | 0.065 | 0.022 | 0.010 |
| 0.01-cat-random-oversample-fixed    | 0.812         | 0.812                   | 0.903    | 0.938      | 0.855 | 0.008 | 0.879 | 0.866 | 0.923 | 0.908 | 0.932 | 0.913 |
| 100-epoch-0.01-cat-smote-fixed      | 0.811         | 0.809                   | 0.900    | 0.929      | 0.845 | 0.012 | 0.900 | 0.865 | 0.930 | 0.900 | 0.925 | 0.899 |
| 100-epoch-0.01-cat-adasyn-fixed     | 0.813         | 0.812                   | 0.920    | 0.922      | 0.851 | 0.012 | 0.885 | 0.865 | 0.921 | 0.914 | 0.922 | 0.917 |
| 0.01-cat--more-augm                 | 0.150         | 0.148                   | 0.759    | 0.140      | 0.157 | 0.000 | 0.006 | 0.031 | 0.016 | 0.018 | 0.264 | 0.111 |
| 0.01-cat-label-smoothing            | 0.808         | 0.807                   | 0.896    | 0.918      | 0.854 | 0.000 | 0.872 | 0.867 | 0.922 | 0.908 | 0.928 | 0.917 |




Notatki:

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
2. Oba ustawienia many augmentations - 100 epok to SPORO za krótko, słaby wynik jest tylko dlatego, że ta augmentacja jest trudna i wymaga więcej epok, można kiedyś powtórzyć eksperyment na 200 epok.

8.   0.01 Cat + label smoothing
9.  0.01 Cat + hierarchicla clustering
10.   0.01 Cat + class weightinh (loss)
13.  0.01 Cat + FLUX.1-dev
14.   0.01 Cat + FLUX.1-schnell
15.    0.01 Cat + SD-3.5 Large Turbo
16. 0.01 Cat + SD-3.5 Large
