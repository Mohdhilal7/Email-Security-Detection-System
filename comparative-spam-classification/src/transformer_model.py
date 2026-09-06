import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from .config import BATCH_SIZE, EPOCHS, GRAD_CLIP_NORM, LEARNING_RATE, MAX_LENGTH, MODELS_DIR, PATIENCE, RANDOM_SEED, RESULTS_DIR, TRANSFORMER_NAME, WARMUP_RATIO, WEIGHT_DECAY
from .data_utils import load_and_prepare, stratified_split
from .evaluation import evaluate_predictions, save_metrics


class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts, self.labels, self.tokenizer, self.max_length = list(texts), list(labels), tokenizer, max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class DistilBertClassifier(nn.Module):
    def __init__(self, model_name=TRANSFORMER_NAME, num_labels=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Linear(hidden, num_labels)
    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(self.dropout(output.last_hidden_state[:, 0, :]))


def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, scheduler, device, train_mode=True):
    model.train(train_mode); total_loss = 0.0; all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    with (torch.enable_grad() if train_mode else torch.no_grad()):
        for batch in tqdm(loader, leave=False):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch); loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step(); scheduler.step()
            total_loss += loss.item()
            all_preds.extend(logits.argmax(dim=1).detach().cpu().numpy()); all_labels.extend(labels.detach().cpu().numpy())
    metrics = evaluate_predictions(all_labels, all_preds); metrics["loss"] = total_loss / max(1, len(loader)); return metrics


def train_transformer(epochs=EPOCHS, max_train_samples=None, max_eval_samples=None):
    set_seed(); df, audit = load_and_prepare(); train, valid, test = stratified_split(df)
    if max_train_samples:
        train = train.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), max_train_samples // 2), random_state=RANDOM_SEED)).reset_index(drop=True)
    if max_eval_samples:
        for name in ["valid", "test"]:
            obj = locals()[name]
            obj = obj.groupby("label", group_keys=False).apply(lambda x: x.sample(min(len(x), max_eval_samples // 2), random_state=RANDOM_SEED)).reset_index(drop=True)
            if name == "valid": valid = obj
            else: test = obj
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)
    train_loader = DataLoader(SpamDataset(train.text_transformer, train.label, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(SpamDataset(valid.text_transformer, valid.label, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE)
    test_loader = DataLoader(SpamDataset(test.text_transformer, test.label, tokenizer, MAX_LENGTH), batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * WARMUP_RATIO), num_training_steps=total_steps)
    best_f1, patience_counter, history = -1.0, 0, []
    start_total = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, scheduler, device, True)
        valid_metrics = run_epoch(model, valid_loader, optimizer, scheduler, device, False)
        history.append({"epoch": epoch, "train_loss": train_metrics["loss"], "train_f1": train_metrics["f1"], "val_loss": valid_metrics["loss"], "val_f1": valid_metrics["f1"]})
        print(f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f} val_loss={valid_metrics['loss']:.4f} val_f1={valid_metrics['f1']:.4f}")
        if valid_metrics["f1"] > best_f1:
            best_f1, patience_counter = valid_metrics["f1"], 0
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model.encoder.save_pretrained(MODELS_DIR / "distilbert_spam")
            tokenizer.save_pretrained(MODELS_DIR / "distilbert_spam")
            torch.save(model.classifier.state_dict(), MODELS_DIR / "distilbert_classifier.pt")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE: break
    total_time = time.perf_counter() - start_total
    best_model = DistilBertClassifier().to(device)
    best_model.encoder = AutoModel.from_pretrained(MODELS_DIR / "distilbert_spam")
    best_model.classifier.load_state_dict(torch.load(MODELS_DIR / "distilbert_classifier.pt", map_location=device))
    test_metrics = run_epoch(best_model, test_loader, optimizer, scheduler, device, False)
    test_metrics.update({"model": "DistilBERT", "training_time_seconds": total_time})
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(RESULTS_DIR / "transformer_history.csv", index=False)
    save_metrics([test_metrics], RESULTS_DIR / "metrics_transformer.csv")
    (RESULTS_DIR / "transformer_run_info.json").write_text(json.dumps({"device": str(device), "model": TRANSFORMER_NAME, "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY, "epochs_requested": epochs, "epochs_completed": len(history), "best_validation_f1": best_f1, "training_time_seconds": total_time}, indent=2), encoding="utf-8")
    return [test_metrics], history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--epochs", type=int, default=EPOCHS); parser.add_argument("--max_train_samples", type=int, default=None); parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args(); rows, _ = train_transformer(args.epochs, args.max_train_samples, args.max_eval_samples); print(pd.DataFrame(rows).to_string(index=False))
