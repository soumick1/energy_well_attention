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
        self.output_dim = output_dim

        if method == "gaussian":
            self.attention = GaussianEnergyWellAttention(embedding_dim)
        elif method == "softmax_exponential":
            self.attention = SoftmaxExponentialEnergyWellAttention(embedding_dim)
        else:
            self.attention = nn.MultiheadAttention(embedding_dim, num_heads=1)

        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.embedding_dim)
        query, key, value = x, x, x
        attention_output, _ = self.attention(query, key, value)
        pooled_output = attention_output.mean(dim=1)
        return self.fc(pooled_output)
