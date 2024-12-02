# SD-Imbalance

This repository is designed for experiments on how to use diffusion-generated synthetic data to address the problem of imbalanced datasets.

# Installation

Install all the required packages by running the following command:
```
pip install -r requirements.txt
```

# Usage
To train the model, run the following command:
```
python train.py
```

To downsample the cats class to 10% of the original size, run the following command:
```
python main.py downsample_class="cat" downsample_ratio=0.1 epochs=5
```
