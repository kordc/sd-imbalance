| Method Name         | Test Accuracy | Test Balanced Accuracy | Airplane | Automobile | Bird | Cat | Deer | Dog | Frog | Horse | Ship | Truck |
|---------------------|---------------|------------------------|----------|------------|------|-----|------|-----|------|-------|------|-------|
| Full data           | 0.855         | 0.853                  | 0.870    | 0.921      | 0.820 | 0.707 | 0.865 | 0.778 | 0.897 | 0.868 | 0.924 | 0.904 |
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
3. Full + many augmentations
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
    - name: ColorJitter
      params:
          brightness: 0.5
          hue: 0.3
    - name: RandomPerspective
      params:
          distortion_scale: 0.2
          p: 1.0
          fill: 0
    - name: RandomRotation
      params:
          degrees: 180
          fill: 0
    - name: RandomSolarize
      params:
          threshold: 0.75
          p: 0.5
5. 0.01 Cat + many augmentations:
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
    - name: ColorJitter
      params:
          brightness: 0.5
          hue: 0.3
    - name: RandomPerspective
      params:
          distortion_scale: 0.2
          p: 1.0
          fill: 0
    - name: RandomRotation
      params:
          degrees: 180
          fill: 0
    - name: RandomSolarize
      params:
          threshold: 0.75
          p: 0.5
6.  0.01 Cat + naive oversampling
7.   0.01 Cat + naive undersampling
8.   0.01 Cat + label smoothing
9.  0.01 Cat + hierarchicla clustering
10.   0.01 Cat + class weightinh (loss)
11.    0.01 Cat + smote
12.  0.01 Cat + FLUX.1-dev
13.   0.01 Cat + FLUX.1-schnell
14.    0.01 Cat + SD-3.5 Large Turbo
15. 0.01 Cat + SD-3.5 Large
