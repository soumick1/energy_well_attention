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

from models import *
from utils import *

dataset_list = ["MNIST", "Fashion-MNIST", "CIFAR-10", "CIFAR-100"]

results = run_experiment_on_datasets(dataset_list)
for result in results:
    print(f"\nDataset: {result['Dataset']}")
    for method, metrics in result.items():
        if method != "Dataset":
            print(f"{method} -> Accuracy: {metrics['Accuracy']:.4f}, Precision: {metrics['Precision']:.4f}, F1 Score: {metrics['F1 Score']:.4f}")
