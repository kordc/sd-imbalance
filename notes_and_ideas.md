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

TODO:
- Spróbować więcej wag label smoothing
- Naprawić more augmentations, żeby nie było tak agresywne
- 0.01 Cat + SD-3.5 Large
- 0.01 Cat + SD-3.5 Large Turbo
- 0.01 Cat + FLUX.1-schnell
- 0.01 Cat + FLUX.1-dev
- Poprawić baseline tak, żeby test set miał >90% accuracy
- Najlepszy z powyższych zbiorów z active learningiem (3 różne funkcje aquisition wyboru)
- Downsampled + black-forest-labs/FLUX.1-Redux-dev jako agresywna augmentacja
- Downsampled + najlepszy z powyższych modeli dotrenowany LoRA do fotorealizmu (albo maszyna z większym GPU, która uciągnie obecną LoRA do FLUX.1-dev)
- Powyższe, ale z Active generation i podobnym mechanizmem
- Powyższe, ale z Active generation i promptem dający guiding - prompt wybierany patrząc na obecny stan sieci na validzie
- Powyższe, ale z active generation i guding z złych obrazków i ich gradientów?
- Powyższe, ale z active generation i LoRA z obencych obrazków
- Wygenerowanie mniejszego datasetu 32x32 - czy to możliwe jak da się więcej kroków denoisingu?
- Train the model data selected through Clustering-based sampling


