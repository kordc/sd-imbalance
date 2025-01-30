# ImbalanceSD

This repository is designed for experiments on how to use diffusion-generated synthetic data to address the problem of imbalanced datasets.

## Installation

Install all the required packages by running the following command:
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
uv run train.py
```

## Usage
### Training
To train the model, run the following command:
```sh
python train.py
```

To downsample the cats class to 10% of the original size, run the following command:
```sh
python train.py downsample_class="cat" downsample_ratio=0.1 epochs=5
```

For other options check out the config/config.yaml file
