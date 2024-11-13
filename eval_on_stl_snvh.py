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
import numpy as np


class GaussianEnergyWellAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(GaussianEnergyWellAttention, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, query, key, value):
        distances = torch.cdist(query, key, p=2)
        weights = torch.exp(-self.alpha * distances ** 2)
        weights = F.softmax(weights, dim=-1)
        return torch.matmul(weights, value), weights


class SoftmaxExponentialEnergyWellAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SoftmaxExponentialEnergyWellAttention, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5), requires_grad=True)

    def forward(self, query, key, value):
        distances = torch.cdist(query, key, p=2)
        weights = torch.exp(-self.alpha * distances)
        weights = F.softmax(weights, dim=-1)
        return torch.matmul(weights, value), weights

class AttentionComparisonModel(nn.Module):
    def __init__(self, embedding_dim, output_dim, method="gaussian"):
        super(AttentionComparisonModel, self).__init__()
        self.method = method
        self.embedding_dim = embedding_dim
        if method == "gaussian":
            self.attention = GaussianEnergyWellAttention(embedding_dim)
        elif method == "softmax_exponential":
            self.attention = SoftmaxExponentialEnergyWellAttention(embedding_dim)
        else:
            self.attention = nn.MultiheadAttention(embedding_dim, num_heads=1)

        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        batch_size = x.size(0)

        if self.method in ["gaussian", "softmax_exponential"]:
            x = x.view(batch_size, -1, self.embedding_dim)
            query, key, value = x, x, x
            attention_output, _ = self.attention(query, key, value)
            pooled_output = attention_output.mean(dim=1)
        else:
            x = x.view(batch_size, -1, self.embedding_dim).permute(1, 0, 2)
            query, key, value = x, x, x
            attention_output, _ = self.attention(query, key, value)
            pooled_output = attention_output.mean(dim=0)

        return self.fc(pooled_output)


def get_data_loaders(dataset_name, batch_size=64):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    if dataset_name == "STL-10":
        train_dataset = datasets.STL10(root='./data', split='train', download=True, transform=transform)
        test_dataset = datasets.STL10(root='./data', split='test', download=True, transform=transform)
        output_dim, embedding_dim = 10, 96*96*3
    elif dataset_name == "SVHN":
        train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
        test_dataset = datasets.SVHN(root='./data', split='test', download=True, transform=transform)
        output_dim, embedding_dim = 10, 32*32*3

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, embedding_dim, output_dim

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
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = 100 * (np.array(y_true) == np.array(y_pred)).sum() / len(y_true)
    precision = precision_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, precision, f1

def run_experiment_on_datasets(dataset_list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    results = []

    for dataset_name in dataset_list:
        train_loader, test_loader, embedding_dim, output_dim = get_data_loaders(dataset_name)

        gaussian_model = AttentionComparisonModel(embedding_dim, output_dim, method="gaussian").to(device)
        gaussian_optimizer = optim.Adam(gaussian_model.parameters(), lr=0.001)
        gauss_accuracy, gauss_precision, gauss_f1 = train_and_evaluate(gaussian_model, train_loader, test_loader, gaussian_optimizer, criterion, device)

        softmax_exponential_model = AttentionComparisonModel(embedding_dim, output_dim, method="softmax_exponential").to(device)
        softmax_exponential_optimizer = optim.Adam(softmax_exponential_model.parameters(), lr=0.001)
        softmax_accuracy, softmax_precision, softmax_f1 = train_and_evaluate(softmax_exponential_model, train_loader, test_loader, softmax_exponential_optimizer, criterion, device)

        conventional_model = AttentionComparisonModel(embedding_dim, output_dim, method="self_attention").to(device)
        conventional_optimizer = optim.Adam(conventional_model.parameters(), lr=0.001)
        conv_accuracy, conv_precision, conv_f1 = train_and_evaluate(conventional_model, train_loader, test_loader, conventional_optimizer, criterion, device)

        results.append({
            "Dataset": dataset_name,
            "Gaussian (Acc, Prec, F1)": (gauss_accuracy, gauss_precision, gauss_f1),
            "Softmax Exponential (Acc, Prec, F1)": (softmax_accuracy, softmax_precision, softmax_f1),
            "Self-Attention (Acc, Prec, F1)": (conv_accuracy, conv_precision, conv_f1),
        })

    return results

dataset_list = ["STL-10", "SVHN"]
results = run_experiment_on_datasets(dataset_list)

for result in results:
    print(f"\nDataset: {result['Dataset']}")
    print(f"  Gaussian Energy Well Attention - Accuracy: {result['Gaussian (Acc, Prec, F1)'][0]:.2f}%, Precision: {result['Gaussian (Acc, Prec, F1)'][1]:.2f}, F1 Score: {result['Gaussian (Acc, Prec, F1)'][2]:.2f}")
    print(f"  Softmax Exponential Attention - Accuracy: {result['Softmax Exponential (Acc, Prec, F1)'][0]:.2f}%, Precision: {result['Softmax Exponential (Acc, Prec, F1)'][1]:.2f}, F1 Score: {result['Softmax Exponential (Acc, Prec, F1)'][2]:.2f}")
    print(f"  Self-Attention - Accuracy: {result['Self-Attention (Acc, Prec, F1)'][0]:.2f}%, Precision: {result['Self-Attention (Acc, Prec, F1)'][1]:.2f}, F1 Score: {result['Self-Attention (Acc, Prec, F1)'][2]:.2f}")
