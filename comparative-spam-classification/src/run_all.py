import argparse
from .config import EPOCHS
from .traditional_ml import train_traditional
from .transformer_model import train_transformer
from .visualize import make_plots


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()
    print("=== PART A: Traditional ML ===")
    train_traditional(max_samples=args.max_train_samples)
    print("=== PART B: DistilBERT / PyTorch ===")
    train_transformer(args.epochs, args.max_train_samples, args.max_eval_samples)
    print("=== VISUALIZATION ===")
    make_plots()
    print("Complete. Check results/ and update the technical report.")

if __name__ == "__main__":
    main()
