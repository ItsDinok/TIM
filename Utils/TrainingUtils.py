import random
from Utils.Evaluation import evaluate_teg_system
from torch.utils.data import DataLoader
import numpy as np
import torch

__all__ = ["run_baseline_evaluation", "remap_labels", "sample_replay", "forward_task_aware", "update_replay_buffer", "no_replay_batch_step"]
# TODO: Make replay buffer a distinct and modular class

def run_baseline_evaluation(model, teg, test_tasks, task_label_maps, criterion, device, n_tasks):
    """
    Runs the baseline evaluation for a model. Used in FWT
    arguments:
        - model: PyTorch model, the main model
        - teg: a TEG module
        - test_tasks: a list of tasks to evaluate on
        - task_label_maps: a dictionary mapping task_id to label
        - criterion: a loss function
        - device: torch device
        - n_tasks: the number of tasks
    returns:
        - baseline_accuracy: a dictionary mapping task_id to baseline accuracy
    """
    model.eval()
    baseline_accuracy = {}

    for task_id in range(n_tasks):
        loader = DataLoader(test_tasks[task_id], batch_size = 512, shuffle = False)
        metrics = evaluate_teg_system(model, teg, loader, criterion, task_label_maps = task_label_maps, device = device)

        baseline_accuracy[task_id] = metrics['top1_acc']

    return baseline_accuracy


def remap_labels(targets, label_map, device):
    """
    Maps global labels to task label spaces
    arguments:
        targets: items to be relabelled
        label_map: a dictionary mapping task_id to label
        - device: torch device
    """
    return torch.tensor(
        [label_map[t.item()] for t in targets],
        device = device,
        dtype = torch.long
    )


def no_replay_batch_step(model, inputs, targets, task_ids, label_map, device):
    """
    Give the model data without using a replay buffer
    arguments:
        - model: a PyTorch model
        - inputs: a tensor of inputs
        - targets: a tensor of targets
        - task_id: the identity of the current task
        - label_map: a dictionary mapping task_id to label
        - device: torch device

    returns:
        - outputs: results from the model
        - targets: predictions
    """
    inputs = inputs.to(device)
    targets = targets.to(device)
    targets = remap_labels(targets, label_map, device)

    task_ids = [t.item() if isinstance(t, torch.Tensor) else t for t in task_ids]
    outputs = forward_task_aware(model, inputs, task_ids)
    return outputs, targets


def sample_replay(replay_buffer, task_label_maps, device, max_samples = 32):
    """
    Feeds samples back into training data

    arguments:
        - replay buffer: the replay buffer samples are drawn from
        - task_label_maps: a dictionary mapping task_id to label
        - device: torch device
        - max_samples: the maximum number of samples to return

    returns:
        - batch_x: a tensor of samples
        - batch_y: a tensor of labels
        - batch_task_ids: a list of task ids
    """
    batch_x, batch_y, batch_task_ids = [], [], []

    for prev_task_id, samples in replay_buffer.items():
        if len(samples) == 0:
            continue

        idxs = np.random.choice(
            len(samples),
            size = min(max_samples, len(samples)),
            replace = False
        )

        prev_map = task_label_maps[prev_task_id]

        for i in idxs:
            x, y = samples[i]
            y = int(y)

            if y not in prev_map:
                continue

            batch_x.append(x.to(device))
            batch_y.append(prev_map[y])
            batch_task_ids.append(prev_task_id)

    return batch_x, batch_y, batch_task_ids


# TODO: Figure out what the fuck this does and give it a docstring
def forward_task_aware(model, batch_x, batch_task_ids):
    # Group indicies by task
    task_groups = {}
    for i, tid in enumerate(batch_task_ids):
        key = str(tid.item() if isinstance(tid, torch.Tensor) else tid)
        task_groups.setdefault(key, []).append(i)

    # Allocate output tensor
    # Infer n_classes from forward pass
    outputs = [None] * len(batch_task_ids)

    for task_key, indices in task_groups.items():
        x = torch.stack([batch_x[i] for i in indices])
        out = model(x, task = task_key)
        for j, original_idx in enumerate(indices):
            outputs[original_idx] = out[j]

    return torch.stack(outputs, dim = 0)


def update_replay_buffer(train_task, buffer_size = 200):
    """
    Update the replay buffer

    arguments:
        - train_task: a task to train on
        - buffer_size: the size of the replay buffer

    returns:
        - samples: a list of samples
    """
    samples = [(x, int(y)) for x, y in train_task]

    if len(samples) > buffer_size:
        samples = random.sample(samples, buffer_size)

    return samples