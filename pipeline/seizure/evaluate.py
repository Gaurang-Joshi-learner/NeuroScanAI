"""
pipeline/seizure/evaluate.py
-----------------------------
Generates full evaluation report: metrics, confusion matrix,
ROC curve, PR curve, and saves plots to disk.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score
)
import json


def plot_confusion_matrix(cm, save_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ['Normal', 'Seizure']
    ax.set(xticks=[0,1], yticks=[0,1],
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True',
           title='Confusion Matrix')
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{cm[i,j]}\n({cm[i,j]/total*100:.1f}%)',
                    ha='center', va='center',
                    color='white' if cm[i,j] > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#2563eb', lw=2, label=f'ROC AUC = {roc_auc:.3f}')
    ax.plot([0,1],[0,1], 'k--', lw=1)
    ax.set(xlabel='False Positive Rate', ylabel='True Positive Rate',
           title='ROC Curve — Seizure Detection', xlim=[0,1], ylim=[0,1.05])
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return roc_auc


def plot_pr_curve(y_true, y_probs, save_path):
    prec, rec, _ = precision_recall_curve(y_true, y_probs)
    ap = average_precision_score(y_true, y_probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, color='#16a34a', lw=2, label=f'PR AUC = {ap:.3f}')
    ax.set(xlabel='Recall', ylabel='Precision',
           title='Precision-Recall Curve — Seizure Detection')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return ap


def plot_training_history(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history['train_loss'], label='Train', color='#2563eb')
    axes[0].plot(history['val_loss'],   label='Val',   color='#dc2626')
    axes[0].set(title='Loss', xlabel='Epoch', ylabel='Loss')
    axes[0].legend()
    axes[1].plot(history['val_auc'], color='#16a34a')
    axes[1].set(title='Validation ROC-AUC', xlabel='Epoch', ylabel='AUC')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def generate_report(y_true, y_probs, model_name, save_dir, history=None):
    os.makedirs(save_dir, exist_ok=True)
    y_pred = (y_probs >= 0.5).astype(int)

    report = classification_report(y_true, y_pred,
                                   target_names=['Normal','Seizure'],
                                   output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    roc_auc = plot_roc_curve(y_true, y_probs,
                              os.path.join(save_dir, f'{model_name}_roc.png'))
    pr_auc  = plot_pr_curve(y_true, y_probs,
                             os.path.join(save_dir, f'{model_name}_pr.png'))
    plot_confusion_matrix(cm, os.path.join(save_dir, f'{model_name}_cm.png'))
    if history:
        plot_training_history(history,
                              os.path.join(save_dir, f'{model_name}_history.png'))

    metrics = {
        'model': model_name,
        'roc_auc': round(roc_auc, 4),
        'pr_auc':  round(pr_auc, 4),
        'seizure_precision': round(report['Seizure']['precision'], 4),
        'seizure_recall':    round(report['Seizure']['recall'], 4),
        'seizure_f1':        round(report['Seizure']['f1-score'], 4),
        'accuracy':          round(report['accuracy'], 4),
        'confusion_matrix':  cm.tolist(),
    }

    with open(os.path.join(save_dir, f'{model_name}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f'\n{model_name} Metrics:')
    for k, v in metrics.items():
        if k != 'confusion_matrix':
            print(f'  {k}: {v}')
    return metrics
