import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .config import RESULTS_DIR


def save_fig(fig, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_plots():
    df = pd.concat([pd.read_csv(RESULTS_DIR / "metrics_traditional.csv"), pd.read_csv(RESULTS_DIR / "metrics_transformer.csv")], ignore_index=True)
    long_df = df.melt(id_vars=["model"], value_vars=["accuracy", "precision", "recall", "f1"], var_name="Metric", value_name="Score")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=long_df, x="Metric", y="Score", hue="model", ax=ax)
    ax.set_ylim(0, 1.05); ax.set_title("Metric Comparison Across Models"); ax.set_ylabel("Score"); ax.set_xlabel("")
    save_fig(fig, "metric_comparison.png")
    for _, row in df.iterrows():
        cm = [[row["tn"], row["fp"]], [row["fn"], row["tp"]]]
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", cbar=False, xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(f"Confusion Matrix — {row['model']}")
        save_fig(fig, f"confusion_matrix_{row['model'].lower().replace(' ', '_')}.png")
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="model", y="training_time_seconds", ax=ax)
    ax.set_title("Training Time Comparison"); ax.set_xlabel(""); ax.set_ylabel("Training time (seconds)")
    save_fig(fig, "training_time_comparison.png")
    history_path = RESULTS_DIR / "transformer_history.csv"
    if history_path.exists():
        hist = pd.read_csv(history_path)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(hist["epoch"], hist["train_loss"], marker="o", label="Train loss")
        ax.plot(hist["epoch"], hist["val_loss"], marker="o", label="Validation loss")
        ax.set_title("DistilBERT Training Behaviour"); ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-entropy loss"); ax.legend()
        save_fig(fig, "transformer_training_curve.png")
    df.to_csv(RESULTS_DIR / "metrics.csv", index=False)

if __name__ == "__main__":
    make_plots()
