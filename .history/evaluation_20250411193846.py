import evaluate
from transformers.trainer_utils import EvalPrediction
import numpy as np
from constants import ID_TO_LABEL

metric_evaluator = evaluate.load("seqeval")


def compute_metrics(eval_predictions: EvalPrediction) -> dict[str, float]:
    """
    Compute evaluation metrics (precision, recall, f1) for predictions.

    First takes the argmax of the logits to convert them to predictions.
    Then we have to convert both labels and predictions from integers to strings.
    We remove all the values where the label is -100, then pass the results to the metric.compute() method.
    Finally, we return the overall precision, recall, and f1 score.

    Args:
        eval_predictions: Evaluation predictions.

    Returns:
        Dictionary with evaluation metrics. Keys: precision, recall, f1.

    NOTE: You can use `metric_evaluator` to compute metrics for a list of predictions and references.
    """
    # Write your code here.
    predictions, labels = eval_predictions
    predictions = np.argmax(predictions, axis=2)  # Get the predicted class indices
    # Remove ignored index (special tokens)
    true_predictions = [
        [ID_TO_LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for (prediction, label) in zip(predictions, labels)
    ]
    true_labels = [
        [ID_TO_LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for (prediction, label) in zip(predictions, labels)
    ]

    results = metric_evaluator.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }