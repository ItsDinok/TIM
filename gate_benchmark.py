import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, Dataset, DataLoader, Subset
import numpy as np
import basic_gate
import os
import pickle
import gate_preprocessing


def evaluate(model, teg, dataloader, criterion, device = "cuda"):
    # Put teg in eval mode
    teg.eval()

    running_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        fallback_count = 0

        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Get task predictions from teg
            task_logits = teg(inputs)
            probs = F.softmax(task_logits, dim = -1)
            certainty, pred_tasks = torch.max(probs, dim = -1)

            batch_outputs = []

            for i in range(inputs.size(0)):
                task_id = pred_tasks[i].item()
                conf = certainty[i].item()

                # Fallback
                if conf <= 0.7: # TODO: Tweak this value
                    task_id = task_logits.size(-1)
                    fallback_count += 1

                # Run model on sample
                output = model(inputs[i].unsqueeze(0), task = task_id)
                batch_outputs.append(output)

            outputs = torch.cat(batch_outputs, dim = 0)

            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

            # Top 1 accuracy
            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(targets).sum().item()

            # Top 5 accuracy
            _, top5_pred = outputs.topk(5, dim = 1)
            correct_top5 += top5_pred.eq(targets.view(-1, 1)).sum().item()

            total += targets.size(0)

        avg_loss = running_loss / total
        top1_acc = 100.0 * correct_top1 / total
        top5_acc = 100.0 * correct_top5 / total

        print(f"Fallback used {fallback_count} times")

        return {
            "loss": avg_loss,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "fallback_count": fallback_count
        }


# TODO: Move this to the gate file
def train_model(model, teg, n_tasks, device, train_tasks, test_tasks):
    """
    Create and train the Gated ResNet32
    """
    optimiser = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=200)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    for task_id, train_task in enumerate(train_tasks):
        print(f"\n=== Training Task {task_id + 1}/{n_tasks} ===")
        trainloader = DataLoader(
            train_task,
            batch_size = 512,
            shuffle = True,
            num_workers = 4,
            pin_memory = True,
            prefetch_factor = 4
        )
       
        # Train task for 200 epochs
        for epoch in range(200):
            model.train()
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimiser.zero_grad()
                outputs = model(inputs, task = task_id)
                loss = criterion(outputs, targets)
                loss.backward()
                optimiser.step()
            scheduler.step()

        evaluate_tasks(model, teg, test_tasks, criterion, task_id)


def evaluate_tasks(model, teg, test_tasks, criterion, task_id):
    # Evaluate on all seen tasks
    model.eval()
    for prev_task_id in range(task_id + 1):
        testloader = DataLoader(
            test_tasks[prev_task_id],
            batch_size = 512,
            shuffle = False,
            num_workers = 4,
            pin_memory = True,
            prefetch_factor = 4
        )

        metrics = evaluate(model, teg, testloader, criterion)
        print(f"Task {prev_task_id + 1} Eval - Loss: {metrics['loss']:.4f}, "
              f"Top1: {metrics['top1_acc']:.2f}%, Top5: {metrics['top5_acc']:.2f}%")


def main():
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    device = "cuda"

    # Data transforms
    transform_train, transform_test = gate_preprocessing.fetch_transforms()

    # Load CIFAR-100
    trainset = torchvision.datasets.CIFAR100(root = "./data", train = True, download = True, transform = transform_train)
    testset = torchvision.datasets.CIFAR100(root = "./data", train = False, download = True, transform = transform_test)

    # Split into incremental tasks

    train_tasks = gate_preprocessing.split_dataset_by_tasks(trainset, n_tasks = 20)
    test_tasks = gate_preprocessing.split_dataset_by_tasks(testset, n_tasks = 20)

    # Construct task dict
    classes_per_task = 5
    task_map = {str(i): classes_per_task for i in range(len(train_tasks))}

    # Create model
    model = basic_gate.GatedResNet32(task_map).to(device)
   
    # Prepare data
    gate_train = gate_preprocessing.prepare_gate_datasets(train_tasks)
    gate_test  = gate_preprocessing.prepare_gate_datasets(test_tasks)

    # Create TEG
    n_tasks = 20
    teg = basic_gate.task_estimation_gate(
        tasks = n_tasks, device = device, 
        train_data = gate_train, 
        test_data = gate_test
    )

    # Train model
    train_model(model, teg, n_tasks, device, train_tasks, test_tasks)
        

if __name__ == "__main__":
    main()
