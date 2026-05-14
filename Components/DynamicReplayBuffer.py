from collections import defaultdict, deque
import random
import torch


def init_class_buffer(max_per_class = 50):
    # Each class get sits own deque
    return defaultdict(lambda: deque(maxlen=max_per_class))


def update_rotating_buffer(replay_buffer, inputs, targets, label_map):
    for x, y in zip(inputs, targets):
        y_mapped = int(label_map[int(y.item())])
        replay_buffer[y_mapped].append((x.cpu(), y_mapped))


def sample_replay(replay_buffer, max_samples, device):
    per_class = max(1, max_samples // len(replay_buffer))
    samples = []

    for label, buffer in replay_buffer.items():
        samples.extend(random.sample(list(buffer), min(per_class, len(buffer))))

    random.shuffle(samples)
    samples = samples[:max_samples]

    batch_x = torch.stack([s[0] for s in samples]).to(device)
    batch_y = torch.tensor([s[1] for s in samples], device=device)
    batch_task_ids = [s[2] for s in samples]  # preserve original task provenance

    return batch_x, batch_y, batch_task_ids