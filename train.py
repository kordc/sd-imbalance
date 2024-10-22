import torch
import torch.optim as optim
import torch.nn as nn
from omegaconf import DictConfig
import hydra
from data import get_dataloader
from model import ResNetModel


@hydra.main(config_path="configs", config_name="default")
def train(config: DictConfig):
    # Set seeds for reproducibility
    torch.manual_seed(config.train.seed)

    # Initialize data and model
    dataloader = get_dataloader(config)
    model = ResNetModel(config)
    model.to(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = getattr(optim, config.train.optimizer)(
        model.parameters(), lr=config.train.lr)

    # Training loop
    model.train()
    for epoch in range(config.train.epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f"Epoch {epoch+1}/{config.train.epochs}, Loss: {running_loss / len(dataloader)}")

    # Evaluate model TODO: Implement evaluation
    model.eval()

    # Save model
    torch.save(model.state_dict(), 'model.pth')

    # Create a metric raport TODO: Implement metric reporting


if __name__ == "__main__":
    train()
