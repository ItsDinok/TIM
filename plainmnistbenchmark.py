import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision.models import resnet18
import numpy as np
import random
from Utils.Evaluation import compute_cl_metrics


class ResNet32(nn.Module):
    """Standard single-head ResNet for CIL baseline. No task awareness."""
    def __init__(self, num_classes):
        super().__init__()
        backbone = resnet18(weights=None)
        # Replace first conv to handle 32x32 input instead of 224x224
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        # Replace final fc with correct output size
        backbone.fc = nn.Linear(backbone.fc.in_features, num_classes)
        self.net = backbone

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_model_single_head(model, n_tasks, epochs, device, train_tasks, test_tasks, global_label_map):
    """
    Standard CIL baseline: single-head ResNet with replay, no task identity.
    Uses a global label space across all tasks rather than per-task heads.

    arguments:
        - model: ResNet32 with num_classes = total classes across all tasks
        - n_tasks: number of tasks
        - epochs: epochs per task
        - device: torch device
        - train_tasks: list of training datasets
        - test_tasks: list of test datasets
        - global_label_map: dict mapping original digit -> contiguous global index
                            e.g. {0:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8}
    """
    replay_buffer = {}
    buffer_size_per_task = 200
    results = {}
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    baseline_accuracy = _single_head_baseline(model, test_tasks, global_label_map, criterion, device, n_tasks)

    for task_id, train_task in enumerate(train_tasks):
        print(f"Task {task_id + 1} / {n_tasks} (SINGLE HEAD)")

        trainloader = DataLoader(train_task, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
        optimiser = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

        for epoch in range(epochs):
            model.train()

            for inputs, targets, _ in trainloader:
                inputs = inputs.to(device)
                batch_x, batch_y = [], []

                # Current task — remap to global label space
                for x, y in zip(inputs, targets):
                    y = int(y.item())
                    if y not in global_label_map:
                        continue
                    batch_x.append(x)
                    batch_y.append(global_label_map[y])

                # Replay — already in global label space
                if replay_buffer:
                    for prev_task_id, samples in replay_buffer.items():
                        if not samples:
                            continue
                        idxs = np.random.choice(len(samples), size=min(32, len(samples)), replace=False)
                        for i in idxs:
                            x, y = samples[i]
                            batch_x.append(x.to(device))
                            batch_y.append(y)  # already remapped when stored

                if not batch_x:
                    continue

                batch_x = torch.stack(batch_x).to(device)
                batch_y = torch.tensor(batch_y, device=device)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            scheduler.step()

        # Store replay with global labels already applied
        replay_buffer[task_id] = _build_single_head_buffer(train_task, global_label_map, buffer_size_per_task)
        results[task_id] = _single_head_evaluate(model, test_tasks, global_label_map, criterion, device)

    metrics = compute_cl_metrics(results, baseline_accuracy, n_tasks)
    print(f"BWT: {metrics['bwt']} \nFWT: {metrics['fwt']}")
    return results, metrics


def _build_single_head_buffer(train_task, global_label_map, buffer_size):
    """Builds replay buffer with labels already in global space."""
    samples = []
    for x, y, *_ in train_task:
        y = int(y)
        if y in global_label_map:
            samples.append((x, global_label_map[y]))
    if len(samples) > buffer_size:
        samples = random.sample(samples, buffer_size)
    return samples


def _single_head_baseline(model, test_tasks, global_label_map, criterion, device, n_tasks):
    model.eval()
    baseline = {}
    for task_id in range(n_tasks):
        loader = DataLoader(test_tasks[task_id], batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
        baseline[task_id] = _single_head_evaluate_single(model, loader, global_label_map, device)
    return baseline


def _single_head_evaluate(model, test_tasks, global_label_map, criterion, device):
    model.eval()
    task_results = {}
    for task_id in range(len(test_tasks)):
        loader = DataLoader(test_tasks[task_id], batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
        acc = _single_head_evaluate_single(model, loader, global_label_map, device)
        print(f"Task {task_id + 1} Eval (single head) - Top-1 Accuracy: {acc:.4f}")
        task_results[task_id] = acc
    return task_results


def _single_head_evaluate_single(model, loader, global_label_map, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            batch_x, batch_y = [], []

            for x, y in zip(inputs, targets):
                y = int(y.item())
                if y not in global_label_map:
                    continue
                batch_x.append(x)
                batch_y.append(global_label_map[y])

            if not batch_x:
                continue

            batch_x = torch.stack(batch_x).to(device)
            batch_y = torch.tensor(batch_y, device=device)

            outputs = model(batch_x)
            _, predicted = outputs.max(1)
            correct += predicted.eq(batch_y).sum().item()
            total += batch_y.size(0)

    return 100.0 * correct / max(1, total)