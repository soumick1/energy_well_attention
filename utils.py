import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, f1_score
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from transformers import BertTokenizer, BertModel
import time


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device):
    model.to(device)
    for epoch in range(10):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return accuracy, precision, f1


def get_data_loaders(dataset_name):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if dataset_name in ["MNIST", "Fashion-MNIST"] else
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset_name == "MNIST":
        dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        output_dim, embedding_dim = 10, 28*28
    elif dataset_name == "Fashion-MNIST":
        dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        output_dim, embedding_dim = 10, 28*28
    elif dataset_name == "CIFAR-10":
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        output_dim, embedding_dim = 10, 32*32*3
    elif dataset_name == "CIFAR-100":
        dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
        output_dim, embedding_dim = 100, 32*32*3


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader, embedding_dim, output_dim
