from collections import defaultdict, deque
import random
import torch


def init_class_buffer(max_per_class = 50):
    # Each class get sits own deque
    return defaultdict(lambda: deque(maxlen=max_per_class))


def update_rotating_buffer(replay_buffer, inputs, targets, label_map):
    for x, y in zip(inputs, targets):
        y_mapped = label_map[int(y.item())]
        replay_buffer[y_mapped].append(x.cpu())


def sample_replay(replay_buffer, max_samples, device):
    all_samples = []
    for label, buffer in replay_buffer.items():
        for x in buffer:
            all_samples.append(x)

    samples = random.sample(all_samples, min(max_samples, len(all_samples)))
    batch_x = torch.stack([x for x, y in samples]).to(device)
    batch_y = torch.tensor([y for x, y in samples]).to(device)
    return batch_x, batch_y