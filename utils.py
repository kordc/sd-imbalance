import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score

CIFAR10_CLASSES = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}


def evaluate_model(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * (np.array(all_preds) ==
                      np.array(all_labels)).sum() / len(all_labels)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds) * 100
    return accuracy, balanced_acc
