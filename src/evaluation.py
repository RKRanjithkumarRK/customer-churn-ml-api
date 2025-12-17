import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score
)

# Import your helper to get train/test split
from feature_engineering import split_data

def load_pipeline(path: str = "models/final_model.pkl"):
    """
    Load and return the saved pipeline (preprocessor + model).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"pipeline file not found at {path}")
    pipeline = joblib.load(path)
    return pipeline

def load_test_data(cleaned_csv: str = "data/cleaned_telco_churn.csv"):
    """
    Load cleaned CSV and return X_test, y_test.
    This uses split_data() from feature_engineering, which performs a
    consistent stratified train/test split.
    """
    if not os.path.exists(cleaned_csv):
        raise FileNotFoundError(f"Cleaned CSV not found: {cleaned_csv}")
    df = pd.read_csv(cleaned_csv)

    # Ensure target is numeric 0/1 (defensive)
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Get test split using your existing function
    # x_train, x_test, y_train, y_test
    x_train, x_test, y_train, y_test = split_data(df)
    return  x_test, y_test

def predict_and_probs(pipeline, x):
    """
    Return y_pred (0/1) and y_prob (probability for positive class).
    y_pred uses default 0.5 threshold here.
    """
    # predicted classes
    y_pred = pipeline.predict(x)

    # Try to get probability for positive class (1)
    y_prob = None

    try:
        proba = pipeline.predict_proba(x) # shape (n_samples, n_classes)
        # find index of positive class (usually 1)
        if proba.shape[1] == 1:
            # Some binary estimators return single-column prob for class 1
            y_prob = proba.ravel()
        else:
            # assume columns order [class0, class1]
            y_prob = proba[:, 1]
    except Exception:
        # Some models may not implement predict_proba (rare for our models).
        # Fall back to decision_function (then rescale with sigmoid) if available.
        try:
            df = pipeline.decision_function(x)
            # Convert decision function to probability-like values using sigmoid
            y_prob = 1 / (1 + np.exp(-df))
        except Exception:
            # If we still can't get probabilities, set to None
            y_prob = None
        
    return y_pred, y_prob

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """
    Compute and return a dict with:
      - accuracy, precision, recall, f1
      - confusion_matrix (2x2)
      - roc_auc (if y_prob provided)
      - pr_auc (if y_prob provided)
    """
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)

    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics['roc_auc'] = None
        try:
            metrics['pr_auc'] = average_precision_score(y_true, y_prob)
        except Exception:
            metrics['pr_auc'] = None

    else:
        metrics['roc_auc'] = None
        metrics['pr_auc'] = None
    
    return metrics

def plot_confusion_matrix(cm, labels=['No', 'Yes'], out_path=None):
    """ Plot confusion matrix as heatmap and optionally save to out_path. 
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    # annotate cells
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i, format(cm[i, j], 'd'),
                    ha = "center", va = "center",
                    color = "white" if cm[i,j] > thresh else "black")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches= 'tight')
    plt.close(fig)
    return fig

def plot_roc_curve(y_true, y_prob, out_path=None):
    """ Compute fpr, tpr, plot ROC curve, show AUC, save optional. 
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4)) 
    ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.3f})') 
    ax.plot([0, 1], [0, 1], 'k--', label='Random') 
    ax.set_xlabel('False Positive Rate') 
    ax.set_ylabel('True Positive Rate') 
    ax.set_title('ROC Curve') 
    ax.legend(loc='lower right')
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return fig

def plot_precision_recall_curve(y_true, y_prob, out_path=None):
    """
    Plot precision-recall curve and show average precision or PR-AUC.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f'PR curve (AP = {ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='lower left')
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    return fig

def threshold_analysis(y_true, y_prob, threshols=[0.2,0.3,0.4,0.5,0.6], out_path=None):
    """ For each threshold compute precision, recall, f1, tp, fp, fn, tn. Return a DataFrame. 
    """
    rows = []
    for t in threshols:
        y_pred_t = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_t).ravel()
        precision = precision_score(y_true, y_pred_t, zero_division=0)
        recall = recall_score(y_true, y_pred_t, zero_division=0)
        f1 = f1_score(y_true, y_pred_t, zero_division=0)
        rows.append({
            'threshold': t,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn), 'tn': int(tn)
        })
    df_thresh = pd.DataFrame(rows)
    if out_path:
        df_thresh.to_csv(out_path, index = False)
    return df_thresh

# -------------------------
# Main runner
# -------------------------
def main(
    pipeline_path="models/final_model.pkl",
    cleaned_csv="data/cleaned_telco_churn.csv",
    reports_dir="reports",
    thresholds=[0.2, 0.3, 0.4, 0.5, 0.6]
):
    os.makedirs(reports_dir, exist_ok=True)

    # 1) load pipeline
    pipeline = load_pipeline(pipeline_path)

    # 2) load test data
    X_test, y_test = load_test_data(cleaned_csv)

    # 3) predict labels + probabilities
    y_pred, y_prob = predict_and_probs(pipeline, X_test)

    # If we couldn't get probabilities, warn and continue with only label metrics
    if y_prob is None:
        print("Warning: probability estimates not available. ROC / PR curves and threshold analysis will be skipped.")

    # 4) compute metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)

    # 5) save plots & threshold table (if probabilities exist)
    cm_path = os.path.join(reports_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics['confusion_matrix'], out_path=cm_path)

    if y_prob is not None:
        roc_path = os.path.join(reports_dir, "roc_curve.png")
        pr_path = os.path.join(reports_dir, "pr_curve.png")
        plot_roc_curve(y_test, y_prob, out_path=roc_path)
        plot_precision_recall_curve(y_test, y_prob, out_path=pr_path)

        thresh_df = threshold_analysis(y_test, y_prob, thresholds, out_path=os.path.join(reports_dir, "thresholds.csv"))
    else:
        thresh_df = pd.DataFrame()

    # 6) print a neat summary
    print("\nMODEL EVALUATION SUMMARY")
    print("-----------------------")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    if metrics['pr_auc'] is not None:
        print(f"PR-AUC   : {metrics['pr_auc']:.4f}")

    print("\nConfusion matrix:")
    print(metrics['confusion_matrix'])
    if not thresh_df.empty:
        print("\nSaved threshold analysis to:", os.path.join(reports_dir, "thresholds.csv"))

    print("\nSaved plots to:", reports_dir)
    return {
        "metrics": metrics,
        "thresholds": thresh_df,
        "reports_dir": reports_dir
    }


if __name__ == "__main__":
    main()
