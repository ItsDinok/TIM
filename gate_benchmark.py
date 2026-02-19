import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import basic_gate


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


class GatedResNet(basic_gate.ResNet):
    def __init__(self, block, num_blocks, num_classes_dict = None):
        """
        num_classes_dict: dict mapping task names -> num classes
        """
        super().__init__(block, num_blocks, num_classes = 0) # Not using original construction
        self.heads = nn.ModuleDict()
        if num_classes_dict is not None:
            for task, n_classes in num_classes_dict.items():
                self.heads[str(task)] = nn.Linear(64 * block.expansion, n_classes)

    def forward(self, x, task = None):
        # Shared backbone
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)

        # Choose head
        if task is None or task not in self.heads:
            task = "fallback"
        return self.heads[str(task)](out)

    def add_output_head(self, task):
        model.heads[task] = nn.Linear(64, 8).to(device)


def GatedResNet32(task_dict):
    return GatedResNet(BasicBlock, [5,5,5], task_dict)


def split_dataset_by_classes(dataset, class_order = None, n_tasks = 10):
    """
    Splits dataset into incremental tasks based on classes

    Args:
        dataset: torchvision dataset
        class_order: optional list specifying class order
        n_tasks: number of tasks

    Returns:
        List of subset datasets, one per task
    """

    n_classes = len(dataset.classes)
    classes_per_task = n_classes // n_tasks

    if class_order is None:
        class_order = np.arange(n_classes)
        np.random.shuffle(class_order)

    task_datasets = []
    targets = np.array(dataset.targets)

    for task_id in range(n_tasks):
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task
        task_classes = class_order[start_class:end_class]

        # Get indices for these classes
        indices = np.where(np.isin(targets, task_classes))[0]
        task_subset = Subset(dataset, indices)
        task_datasets.append(task_subset)

    return task_datasets


def evaluate(model, dataloader, criterion, device = "cuda"):
    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            # Top-1 accuracy
            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(targets).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, dim = 1)
            correct_top5 += top5_pred.eq(targets.view(-1, 1)).sum().item()

            total += targets.size(0)

        avg_loss = running_loss / total
        top1_acc = 100.0 * correct_top1 / total
        top5_acc = 100.0 * correct_top5 / total

        return {
            "loss": avg_loss,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc
        }


def relabel_task_subset(subset, task_classes):
    """
    Remap targets of a subset to 0..len(task_classes)-1 for CrossEntropyLoss
    """

    class_to_idx = {cls: i for i, cls in enumerate(task_classes)}
    
    old_targets = np.array(subset.dataset.targets)[subset.indices]
    new_targets = np.array([class_to_idx[t] for t in old_targets])

    # Create a new subset with remapped targets
    subset.dataset.targets = new_targets.tolist()
    return subset


def expand_output_layer(model, n_new_classes):
    old_linear = model.linear
    in_features = old_linear.in_features
    out_features = old_linear.out_features

    new_out_features = out_features + n_new_classes
    new_linear = nn.Linear(in_features, new_out_features).to(old_linear.weight.device)

    # Copy old weights
    new_linear.weight.data[:out_features] = old_linear.weight.data
    new_linear.bias.data[:out_features] = old_linear.bias.data

    model.linear = new_linear


def process_subset(subset, task_id):
    data = torch.stack([x.clone() for x, _ in subset])
    data[:, -1] = task_id
    return data


def main():
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2, saturation = 0.2, hue = 0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761))
    ])

    # Load CIFAR-100
    trainset = torchvision.datasets.CIFAR100(root = "./data", train = True, download = True, transform = transform_train)
    testset = torchvision.datasets.CIFAR100(root = "./data", train = False, download = True, transform = transform_test)

    # Split into incremental tasks
    n_tasks = 10
    train_tasks = split_dataset_by_classes(trainset, n_tasks = n_tasks)
    test_tasks = split_dataset_by_classes(testset, n_tasks = n_tasks)

    # Construct task dict
    task_map = {}
    for i in range(len(train_tasks)):
        task_map[str(i)] = i
    task_map[i] = 100

    # Initialise model
    model = GatedResNet32(task_map).to(device)
    
    gate_train_tasks = [process_subset(subset, i) for i, subset in enumerate(train_tasks)]
    gate_test_tasks = [process_subset(subset, i) for i, subset in enumerate(test_tasks)]

    teg = basic_gate.task_estimation_gate(tasks = n_tasks, device = device, 
        train_data = gate_train_tasks, test_data = gate_test_tasks)

    optimiser = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=200)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    trainloader = DataLoader(train_task, batch_size=2048, shuffle=True, num_workers=4, pin_memory=True, prefetch_factor=4)
    
    for epoch in range (200):
        model.train()
        for inputs, targets, task_name in train_loader:
            inputs, targets = inputs.to(device), outputs.to(device)

            optimiser.zero_grad()

            # Forward pass through correct head
            outputs = model(inputs, task = task_name)

            loss = criterion(outputs, targets)

            # Backprop
            loss.backward()
            optimiser.step()
        print(f"Epoch: {epoch + 1} / {200} done.")

        model.eval()
        for prev_task_id in range(task_id + 1):
            testloader = DataLoader(
                test_tasks[prev_task_id], batch_size=512, shuffle=False, num_workers=4, pin_memory=True, prefetch_factor=4
            )
            metrics = evaluate(model, testloader, criterion)
            print(f"Task {prev_task_id+1} Eval - Loss: {metrics['loss']:.4f}, "
                  f"Top1: {metrics['top1_acc']:.2f}%, Top5: {metrics['top5_acc']:.2f}%")


if __name__ == "__main__":
    main()
