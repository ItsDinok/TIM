import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
# TODO: Extensive commenting. Some areas of code are too dense

__all__ = ["evaluate_teg_system", "evaluate_tasks", "compute_cl_metrics"]

def _get_task_predictions(teg, inputs):
    """
    INTERNAL ONLY
    Gets the task predictions from the TEG module

    arguments:
        - teg: TEG module trained on the same data as the model.
        - inputs: input tensor

    returns:
        - certainty: softmax value approximating model confidence
        - pred_tasks: list of task predictions
    """
    task_logits = teg(inputs)
    probs = F.softmax(task_logits, dim=1)
    certainty, pred_tasks = torch.max(probs, dim=1)
    return certainty, pred_tasks


def _compute_metrics(outputs, targets, criterion, k):
    """
    INTERNAL ONLY
    Computes the metrics of the model

    Arguments:
        - outputs: output tensor (predictions)
        - targets: target tensor (truth)
        - criterion: PyTorch loss function
        - k: top-k prediction

    Returns:
        Dictionary with:
        "top1": number of correct top1 predictions
        "topn": number of correct topn predictions
        "loss": average loss of the model
        "skipped": number of skipped predictions
    """
    loss = 0.0
    count = 0
    skipped = 0
    correct_top1 = 0
    correct_topn = 0

    if outputs.dim() == 1:
        outputs = outputs.unsqueeze(0)
    if targets.dim() == 0:
        targets = targets.unsqueeze(0)
    targets = targets.long()

    # Clamp k to available classes
    k = min(k, outputs.shape[1])

    loss = criterion(outputs, targets).item()

    # Accuracy
    _, predicted = outputs.max(1)
    correct_top1 += predicted.eq(targets).sum().item()

    _, topn_predicted = outputs.topk(k, dim=1)
    correct_topn += topn_predicted.eq(targets.view(-1, 1)).sum().item()

    return {
        "top1": correct_top1,
        "topn": correct_topn,
        "loss": loss,
        "skipped": skipped
    }


def evaluate_teg_system(model, teg, dataloader, criterion, device, task_label_maps, head_size = 5, confidence_threshold = 0.7):
    """
    This function evaluates the performance of a TEG-enabled system over multiple tasks

    arguments:
        - model: PyTorch model. The main model being evaluated.
        - teg: TEG module trained on the same data as the model.
        - dataloader: PyTorch dataloader used for model evaluation.
        - criterion: PyTorch criterion
        - device: PyTorch device
        - task_label_maps: Dictionary of task labels mapped to the output head.
        - head_size: Number of classes considered for top n accuracy
        - confidence_threshold: softmax value below which the sample is routed to a main model's fallback head

    returns:
        - Dictionary with endpoints:
        - "loss" : average loss of the model
        - "top1_acc" : average accuracy of the model
        - "topn_acc" : top n accuracy where n = head size
        - "fallback_count": number of times fallback head used
    """
    teg.eval()
    model.eval()

    # Initial state
    running_loss = 0.0
    correct_top1, correct_topn = 0, 0
    fallback_count, skipped = 0, 0
    total = 0

    with (torch.no_grad()):
        for inputs, targets, task_ids in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Get task predictions from TEG
            certainty, pred_tasks = _get_task_predictions(teg, inputs)

            batch_outputs, batch_targets = [], []

            # Process each sample
            for i in range(inputs.size(0)):
                pred_task = str(pred_tasks[i].item())
                confidence = certainty[i].item()
                true_label = targets[i].item()
                true_task = task_ids[i].item()
                true_task_str = str(true_task)

                x = inputs[i].unsqueeze(0)

                # Handle low confidence / fallbacks
                if confidence <= confidence_threshold:
                    # TODO: Implement this
                    fallback_count += 1
                    continue
                    #output = model(x, task = "fallback")
                    target = true_label
                    fallback_count +=1
                elif pred_task != true_task_str:
                    skipped += 1
                    continue
                else:
                    output = model(x, task = str(pred_task))
                    label_map = task_label_maps[int(true_task)]
                    if true_label not in label_map:
                        skipped += 1
                        continue
                    target = label_map[true_label]

                batch_outputs.append(output)
                batch_targets.append(target)

            if not batch_outputs:
                continue

            outputs = torch.cat(batch_outputs, dim = 0)
            batch_targets = torch.tensor(batch_targets, device = device)

            total += batch_targets.size(0)
            computed = _compute_metrics(outputs, batch_targets, criterion, head_size)
            correct_top1 += computed["top1"]
            correct_topn += computed["topn"]
            running_loss += computed["loss"]

        avg_loss = running_loss / max(1, total)
        top1_accuracy = 100 * correct_top1 / max(1, total)
        topn_accuracy = 100 * correct_topn / max(1, total)

        print(f"Fallback head used: {fallback_count} | Skipped wrong task: {skipped}.")

    return {
        "loss": avg_loss,
        "top1_acc": top1_accuracy,
        "topn_acc": topn_accuracy,
        "fallback_count": fallback_count,
    }


def evaluate_tasks(model, teg, test_tasks, criterion, device, task_label_maps = None):
    """
    Evaluates the model on all seen tasks

    arguments:
        - model: PyTorch model
        - teg: TEG module
        - test_tasks: dict of tasks to evaluate
        - criterion: PyTorch criterion
        - task_label_maps: Dictionary of task labels mapped to the output head.

    returns:
        - Dictionary of task accuracies
    """
    teg.eval()
    model.eval()
    task_results = {}

    for eval_task_id in range(len(test_tasks)):
        testloader = DataLoader(
            test_tasks[eval_task_id],
            batch_size = 512,
            shuffle = False,
            num_workers = 4,
            pin_memory = True,
            prefetch_factor = 2
        )

        if task_label_maps:
            metrics = evaluate_teg_system(model, teg, testloader, criterion, device, task_label_maps)
            print(f"Task {eval_task_id + 1} Eval - Loss: {metrics['loss']:.4f}, "
                  f"Top-1 Accuracy: {metrics['top1_acc']:.4f}, Top-N Accuracy: {metrics['topn_acc']:.4f}")
            task_results[eval_task_id] = float(metrics['top1_acc'])
        else:
            # TODO: Emergency fallback logic
            raise Exception("No task labels provided.")

    return task_results


# NOTE: For BWT and FWT I have sacrificed speed and concurrency for readability and debuggability
# This is intentional
def _calculate_bwt(final_task, results):
    """
    Calculates the backwards transfer metric, defined by:
    BWT = 1 / (tasks - 1) * sum of all task accuracy decreases after final task

    arguments:
        - final_task: index of last task
        - results: Dictionary of task accuracies
    """
    bwt = 0.0
    count = 0

    for i in range(final_task):
        if (
            i in results.get(final_task, {}) and
            i in results.get(i, {})
        ):
            final_accuracy = results[final_task][i]
            initial_accuracy = results[i][i]

            # Safety check to prevent dict bugs
            if isinstance(final_accuracy, (int, float)) and isinstance(initial_accuracy, (int, float)):
                bwt += final_accuracy - initial_accuracy
                count += 1

    bwt = bwt / max(1, count)
    return bwt

# NOTE: For BWT and FWT I have sacrificed speed and concurrency for readability and debuggability
# This is intentional
def _calculate_fwt(baseline_accuracy, results, n_tasks):
    fwt = 0.0
    count = 0

    for i in range(1, n_tasks):
        if (
            i - 1 in results and
            i in results[i - 1] and
            i in baseline_accuracy
        ):
            fwt += results[i - 1][i] - baseline_accuracy[i]
            count += 1

    if count == 0:
        print("WARNING: No valid FWT entries found!")
    fwt = fwt / max(1, count)
    return fwt


def compute_cl_metrics(results, baseline_accuracy, n_tasks):
    """
    Computes standard continual learning metrics:
    - Accuracy Matrix
    - Forward Transfer
    - Backward Transfer

    Arguments:
        - results: Dictionary of task accuracies
        - baseline_accuracy: baseline accuracy
        - n_tasks: number of tasks

    returns:
    Dictionary:
        - "bwt": Backward Transfer Metric
        - "fwt": Forward Transfer Metric
    """
    final_task = n_tasks - 1
    bwt = _calculate_bwt(final_task, results)
    fwt = _calculate_fwt(baseline_accuracy, results, n_tasks)

    return {
        "bwt": bwt,
        "fwt": fwt
    }
