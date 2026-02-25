import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset
import numpy as np

"""
A task estimation gate is a system by which a classifier guesses which task a sample, or batch of samples belongs to.
The idea behind this is to ease the decision burden on CIL systems that can be pretrained and allow them to treat data as 
TIL, enabling the use of multiple output heads.

This is the most basic, least-optimised version of the task estimation gate: a random forest that classifies tasks
"""

# ================================================================================================================
# ================================================================================================================
# Task Estimation Gate 
# ================================================================================================================
# ================================================================================================================

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride = 1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class TaskEstimationGate(nn.Module):
    def __init__(self, block, num_blocks, num_tasks = 10):
        """
        num_tasks: Number of tasks in the output head
        """
        super().__init__()
        self.in_planes = 16
        self.feature_dim = 64 * block.expansion

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.feature_dim, num_tasks)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, task = None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

    def expand_output_layer(self, _classes):
        old_linear = self.fc
        in_features = old_linear.in_features
        out_features = old_linear.out_features

        new_out_features = out_features + n_new_classes
        new_linear = nn.Linear(in_features, new_out_features).to(old_linear.weight.device)

        # Copy old weights
        new_linear.weight.data[:out_features] = old_linear.weight.data
        new_linear.bias.data[:out_features] = old_linear.bias.data

        self.fc = new_linear


def evaluate_teg(teg, dataloader, device = "cuda"):
    """
    This is only to determine the efficacy of a given teg and isn't a full evaluation function
    """
    teg.eval()
    teg.to(device)

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, labels = inputs.to(device), targets.to(device)
            outputs = teg(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy
            


def task_estimation_gate(train_data, test_data, tasks = 10, device = "cuda"):
    model = TaskEstimationGate(BasicBlock, [5, 5, 5], num_tasks = tasks).to(device)

    print("Training task estimation gate")
    optimiser = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=200)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    trainloader = DataLoader(train_data, batch_size = 512, shuffle = True, num_workers = 4, pin_memory = True, prefetch_factor = 4)
    testloader = DataLoader(test_data, batch_size = 512, shuffle = False, num_workers = 3, pin_memory = True, prefetch_factor = 4)

    # Train for 200 epochs on this task
    for epoch in range(50):
        model.train()
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimiser.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimiser.step()
        scheduler.step()
    
    print("Task estimation gate trained")
    print("TEG accuracy:", evaluate_teg(model, testloader)) 

    return model


# ================================================================================================================
# ================================================================================================================
# Gated ResNet
# ================================================================================================================
# ================================================================================================================

class GatedResNet(nn.Module):
    def __init__(self, block, num_blocks, task_map = None, num_classes = 100):
        """
        task_map: dict mapping task names -> num classes
        num_classes: number of classes for fallback head
        """
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride = 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Output heads
        self.heads = nn.ModuleDict()
        if task_map is not None:
            for task, n_classes in task_map.items():
                self.heads[str(task)] = nn.Linear(64 * block.expansion, n_classes)

        # Fallback head
        self.heads["fallback"] = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, task = None):
        # Shared backbone
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        # Choose head
        if task is None or task not in self.heads:
            task = "fallback"
        return self.heads[str(task)](out)

    def add_output_head(self, task, n_classes, in_features = None):
        if in_features is None:
            in_features = self.in_planes # Last layer output features
        device = next(self.parameters()).device
        self.heads[str(task)] = nn.Linear(in_features, n_classes).to(device)

    def expand_fallback_head(self, n_classes):
        old_linear = self.heads["fallback"]
        in_features = old_linear.in_features
        out_features = old_linear.out_features

        new_out_features = out_features + n_classes
        new_linear = nn.Linear(in_features, new_out_features).to(old_linear.weight.device)

        # Copy old weights
        new_linear.weight.data[:out_features] = old_linear.weight.data
        new_linear.bias.data[:out_features] = old_linear.bias.data

        self.heads["fallback"] = new_linear


def GatedResNet32(task_dict):
    """
    This function is needed as the user is not expected to provide their own block definition
    (But they can if they so wish)
    This function is not needed for the gate as the user is not expected to make their own gate
    (But they can if they so wish)
    """
    return GatedResNet(block = BasicBlock, num_blocks = [5,5,5], task_map = task_dict)

