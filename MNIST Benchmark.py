import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from Utils.DataBackends import MNISTBackend
from Utils.Evaluation import *
from Utils.ModelConstruction import *
from Utils.TrainingUtils import *
import plainmnistbenchmark

# TODO: Write clear documentation
# TODO: Fix head size mismatch evaluation issue

def train_model_no_replay(model, teg, n_tasks, epochs, device, train_tasks, test_tasks, task_label_maps):
    results = {}
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
    baseline_accuracy = run_baseline_evaluation(model, teg, test_tasks, task_label_maps, criterion, device = device, n_tasks = n_tasks)

    for task_id, train_task in enumerate(train_tasks):
        print(f"Task {task_id + 1} / {n_tasks} (NO REPLAY)")
        if task_id > 0:
            model.freeze_layers()

        loader = DataLoader(train_task, batch_size = 64, shuffle = True, num_workers = 4, pin_memory = True)
        optimiser = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = 0.1,
            momentum = 0.9,
            weight_decay = 5e-4,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max = epochs)
        label_map = task_label_maps[task_id]

        for epoch in range(epochs):
            model.train()

            for inputs, targets, task_ids in loader:
                outputs, targets = no_replay_batch_step(model, inputs, targets, task_ids, label_map, device)
                loss = criterion(outputs, targets)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            scheduler.step()

        results[task_id] = evaluate_tasks(
            model, teg, test_tasks, criterion, device, task_label_maps = task_label_maps
        )

    metrics = compute_cl_metrics(results, baseline_accuracy, n_tasks)
    print(f"BWT: {metrics['bwt']} \n FWT: {metrics['fwt']}")


def train_model(model, teg, n_tasks, epochs, device, train_tasks, test_tasks, task_label_maps):
    # State
    replay_buffer = {}
    buffer_size_per_task = 200
    results = {}
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)

    # Baseline for FWT
    baseline_accuracy = run_baseline_evaluation(model, teg, test_tasks, task_label_maps, criterion, device, n_tasks)

    # Sequential training
    for task_id, train_task in enumerate(train_tasks):
        print(f"Task {task_id + 1} / {n_tasks} (REPLAY)")
        label_map = task_label_maps[task_id]
        if task_id > 0:
            model.freeze_layers()

        trainloader = DataLoader(train_task, batch_size = 512, shuffle = True, num_workers = 4, pin_memory = True)
        optimiser = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr = 0.1,
            momentum = 0.9,
            weight_decay = 5e-4,
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max = epochs)

        # Train loop
        for epoch in range(epochs):
            model.train()

            for inputs, targets, task_ids in trainloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                batch_x, batch_y, batch_task_ids = [], [], []

                # Current task data
                for x, y in zip(inputs, targets):
                    y = int(y.item())

                    assert y in label_map

                    batch_x.append(x)
                    batch_y.append(label_map[y])
                    batch_task_ids.append(task_id)

                # Replay data
                if replay_buffer:
                    rx, ry, rtask_ids = sample_replay(
                        replay_buffer,
                        task_label_maps,
                        device = device,
                        max_samples = 32
                    )

                    batch_x.extend(rx)
                    batch_y.extend(ry)
                    batch_task_ids.extend(rtask_ids)

                # Skip empty batch
                if len(batch_x) == 0:
                    continue

                batch_x = torch.stack(batch_x).to(device)
                batch_y = torch.tensor(batch_y, device = device)

                # Task aware forward pass
                outputs = forward_task_aware(model, batch_x, batch_task_ids)

                # Loss + optim step
                loss = criterion(outputs, batch_y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            scheduler.step()

        # Update replay buffer
        replay_buffer[task_id] = update_replay_buffer(train_task, buffer_size = buffer_size_per_task)

        # Evaluate
        task_results = evaluate_tasks(model, teg, test_tasks, criterion, device, task_label_maps = task_label_maps)
        results[task_id] = task_results

    # Continual learning metrics
    metrics = compute_cl_metrics(results, baseline_accuracy, n_tasks)
    print(f"BWT: {metrics['bwt']} \nFWT: {metrics['fwt']}")


# TODO: Make config input for number of tasks: hardcoding a short-term solution
def main(buffer = 0):
    device = "cuda"
    backend = MNISTBackend()
    experiment = build_experiment(backend, root = "./data", device = device)

    if buffer == 0:
        train_model(
            experiment["model"],
            experiment["teg"],
            epochs = 50,
            n_tasks = 3,
            device = device,
            train_tasks = experiment["train_tasks"],
            test_tasks = experiment["test_tasks"],
            task_label_maps = experiment["task_label_maps"],
        )
    elif buffer == 1:
        train_model_no_replay(
        experiment["model"],
        experiment["teg"],
        epochs = 50,
        n_tasks = 3,
        device = device,
        train_tasks = experiment["train_tasks"],
        test_tasks = experiment["test_tasks"],
        task_label_maps = experiment["task_label_maps"]
        )
    else:
        single_head_model = plainmnistbenchmark.ResNet32(num_classes=len(experiment["global_label_map"])).to(device)
        plainmnistbenchmark.train_model_single_head(
            single_head_model,
            n_tasks=3,
            epochs=50,
            device=device,
            train_tasks=experiment["train_tasks"],
            test_tasks=experiment["test_tasks"],
            global_label_map=experiment["global_label_map"],
        )



if __name__ == "__main__":
    main(2)
