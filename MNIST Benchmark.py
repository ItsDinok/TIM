import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
from Utils.DataBackends import MNISTBackend
from Utils.Evaluation import *
from Utils.ModelConstruction import *
from Utils.TrainingUtils import *

# TODO: Write clear documentation
# TODO!!!: Ensure TEG is actually doing TEG things and ISN'T AN ORACLE
# TODO: TEG is predicting label not task. This needs to be fixed ASAP
# TODO: This isn't actually doing what it should. Write down formally what it should do and get that working
# TODO: Fix head size mismatch evaluation issue

def train_model_no_replay(model, teg, n_tasks, epochs, device, train_tasks, test_tasks, task_label_maps):
    results = {}
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
    baseline_accuracy = run_baseline_evaluation(model, teg, test_tasks, task_label_maps, criterion, device = device, n_tasks = n_tasks)

    for task_id, train_task in enumerate(train_tasks):
        print(f"Task {task_id + 1} / {n_tasks} (NO REPLAY)")
        model.freeze_layers()

        loader = DataLoader(teg, batch_size = 64, shuffle = True, num_workers = 4, pin_memory = True)
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

            for inputs, targets in loader:
                outputs, targets = no_replay_batch_step(model, inputs, targets, task_id, label_map, device)
                loss = criterion(outputs, targets)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

            scheduler.step()

        results[task_id] = evaluate_tasks(
            model, teg, test_tasks, criterion, task_label_maps = task_label_maps
        )

    metrics = compute_cl_metrics(results, baseline_accuracy, n_tasks)
    print(f"BWT: {metrics['BWT']} \n FWT: {metrics['FWT']}")


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
        model.freeze_layers()

        if task_id > 0:
            model.expand_fallback_head(len(label_map))

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

            for inputs, targets in trainloader:
                batch_x, batch_y, batch_task_ids = [], [], []

                # Current task data
                for x, y in zip(inputs, targets):
                    y = int(y.item())

                    if y not in label_map:
                        continue

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
        task_results = evaluate_tasks(model, teg, test_tasks, criterion, task_label_maps = task_label_maps)
        results[task_id] = task_results

    # Continual learning metrics
    metrics = compute_cl_metrics(results, baseline_accuracy, n_tasks)
    print(f"BWT: {metrics['bwt']} \nFWT: {metrics['fwt']}")


# TODO: Make config input for number of tasks: hardcoding a short-term solution
def main(buffer = True):
    device = "cuda"
    backend = MNISTBackend()
    experiment = build_experiment(backend, root = "./data", device = device)

    if buffer:
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
    else:
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


if __name__ == "__main__":
    main(False)
