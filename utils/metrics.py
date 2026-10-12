import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)


def compute_metrics(labels, preds, scores=None, ids=None):
    """Metrics with Human (label 1) as the positive class."""
    if ids is not None:
        dedup = {}
        for i, sid in enumerate(ids):
            dedup[str(sid)] = i
        order = list(dedup.values())
        labels = [labels[i] for i in order]
        preds = [preds[i] for i in order]
        if scores is not None:
            scores = [scores[i] for i in order]

    y_true = [int(l) for l in labels]
    y_pred = [int(p) for p in preds]

    human_recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    machine_recall = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    metrics = {
        'human_recall': human_recall,
        'machine_recall': machine_recall,
        'acc': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        'recall': human_recall,
        'f1': f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        'auroc': float('nan'),
        'fpr95': float('nan'),
        'aupr': float('nan'),
    }
    if scores is not None and len(set(y_true)) == 2:
        s = np.asarray(scores, dtype=float)
        metrics['auroc'] = roc_auc_score(y_true, s)
        metrics['aupr'] = average_precision_score(y_true, s)
        fpr, tpr, _ = roc_curve(y_true, s)
        mask = tpr >= 0.95
        metrics['fpr95'] = float(fpr[np.argmax(mask)]) if mask.any() else 1.0
    return metrics


def print_metrics(metrics):
    print(f"HumanRec: {metrics['human_recall']:.4f}  MachineRec: {metrics['machine_recall']:.4f}")
    print(f"ACC: {metrics['acc']:.4f}  Precision: {metrics['precision']:.4f}  "
          f"Recall: {metrics['recall']:.4f}  F1: {metrics['f1']:.4f}")
    print(f"AUROC: {metrics['auroc']:.4f}  FPR@95%TPR: {metrics['fpr95']:.4f}  "
          f"AUPR: {metrics['aupr']:.4f}")
