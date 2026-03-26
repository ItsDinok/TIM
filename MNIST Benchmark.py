import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import basic_gate
import get_mnist
from get_mnist import MultiTaskDataset


def evaluate(model, teg, dataloader, criterion, device = "cuda"):
    # Put teg in eval mode
    teg.eval()

    running_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
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
            _, top3_pred = outputs.topk(3, dim = 1)
            correct_top3 += top3_pred.eq(targets.view(-1, 1)).sum().item()

            total += targets.size(0)

        avg_loss = running_loss / total
        top1_acc = 100.0 * correct_top1 / total
        top5_acc = 100.0 * correct_top3 / total

        print(f"Fallback used {fallback_count} times")

        return {
            "loss": avg_loss,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "fallback_count": fallback_count
        }


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


def train_model(model, teg, n_tasks, epochs, device, train_tasks, test_tasks):
    """
    Create and train gated RN32
    """
    replay_buffer = {}
    buffer_size_per_task = 200  # Tweakable

    for task_id, train_task in enumerate(train_tasks):
        print(f"\n=== Training Task {task_id + 1}/{n_tasks} ===")
        trainloader = DataLoader(
            train_task,
            batch_size=512,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=4
        )

        if task_id > 0:
            model.freeze_layers()
            model.expand_fallback_head(len(train_task.dataset.classes))

        optimiser = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=200)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Train task for 100 epochs
        for epoch in range(200):
            model.train()
            for inputs, targets in trainloader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Replay buffer antics
                if replay_buffer:
                    replay_inputs = []
                    replay_targets = []
                    for prev_task_id, samples in replay_buffer.items():
                        idx = np.random.choice(len(samples), size=32, replace=False)
                        for i in idx:
                            replay_inputs.append(samples[i][0])
                            replay_targets.append(samples[i][1])
                    if replay_inputs:
                        replay_inputs = torch.stack(replay_inputs).to(device)
                        replay_targets = torch.tensor(replay_targets).to(device)
                        # Concatenate with current branch
                        inputs = torch.cat([inputs, replay_inputs], dim=0)
                        targets = torch.cat([targets, replay_targets], dim=0)

                optimiser.zero_grad()
                outputs = model(inputs, task=task_id)
                loss = criterion(outputs, targets)
                loss.backward()
                optimiser.step()
            scheduler.step()

        # Store samples in buffer
        all_samples = [(x, y) for x, y in train_task]
        if len(all_samples) > buffer_size_per_task:
            all_samples = random.sample(all_samples, buffer_size_per_task)
        replay_buffer[task_id] = all_samples

        evaluate_tasks(model, teg, test_tasks, criterion, task_id)



def main():
    np.random.seed(42)
    torch.backends.cudnn.benchmark = True
    device = "cuda" # No if statement because a ResNet shouldn't run on CPU

    trainset, testset = get_mnist.load_mnist(root = "./data")

    task_bundles, global_map = get_mnist.build_mnist_cil_tasks(
        root = "./data",
        train_transform = trainset.transform,
        test_transform = testset.transform,
        use_local_labels = False,
        download = True,
    )

    main_train = ConcatDataset([bundle.train for bundle in task_bundles])
    main_test = ConcatDataset([bundle.test for bundle in task_bundles])

    main_train_loader = DataLoader(main_train, batch_size = 64, shuffle = True, num_workers = 2)
    main_test_loader = DataLoader(main_test, batch_size = 64, shuffle = False, num_workers = 2)

    # Teg data
    gate_train_tasks = get_mnist.split_dataset_by_tasks(trainset)
    gate_test_tasks = get_mnist.split_dataset_by_tasks(testset)

    gate_train = get_mnist.prepare_gate_datasets(gate_train_tasks)
    gate_test = get_mnist.prepare_gate_datasets(gate_test_tasks)

    gate_train_loader = DataLoader(gate_train, batch_size = 64, shuffle = True, num_workers = 2)
    gate_test_loader = DataLoader(gate_test, batch_size = 64, shuffle = False, num_workers = 2)

    # Sanity checks
    labels = [label for _, label in main_train]
    unique_labels = np.unique(labels)
    print("Main model labels:", unique_labels)

    for x, y in main_train_loader:
        print("Main batch x:", x.shape)
        print("Main batch class labels:", y[:10])
        break

    for x, task_id in gate_train_loader:
        print("Gate batch x:", x.shape)
        print("Gate batch task ids:", task_id[:10])
        break

    # Model setup
    classes_per_task = 3
    task_map = {str(i): classes_per_task for i in range(len(task_bundles))}
    model = basic_gate.GatedResNet32(task_map).to(device)

    teg = basic_gate.task_estimation_gate(
        tasks = len(gate_train_tasks),
        device = device,
        train_data = gate_train,
        test_data = gate_test
    )

    train_tasks = [bundle.train for bundle in task_bundles]
    test_tasks = [bundle.train for bundle in task_bundles]

    train_model(model, teg, epochs = 50, n_tasks = 3, device = device, train_tasks = train_tasks, test_tasks = test_tasks)


if __name__ == "__main__":
    main()
