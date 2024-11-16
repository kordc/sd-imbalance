# SD-Imbalance

This project aims to address class imbalance in image classification tasks using deep learning models. The project is built using PyTorch and Hydra for configuration management.

## Project Structure

- `configs/`: Contains configuration files for different aspects of the project.
- `data.py`: Handles data loading and preprocessing.
- `model.py`: Defines the model architecture.
- `train.py`: Contains the training and validation logic.
- `utils.py`: Utility functions for metrics calculation, checkpoint saving, and metrics reporting.

## Configuration

The project uses Hydra for configuration management. The main configuration file is `configs/default.yaml`, which includes other configuration files:

- `augmentation.yaml`: Configuration for data augmentation.
- `dataset.yaml`: Configuration for dataset parameters.
- `model.yaml`: Configuration for model parameters.

### Example Configuration (`configs/default.yaml`)