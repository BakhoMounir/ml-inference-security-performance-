"""
Independent accuracy evaluation for the dynamically-quantized (INT8) model.

Mirrors src/evaluate.py exactly, so the FP32 and INT8 accuracy numbers are
produced by the identical evaluation loop -- the only thing that differs
between this script and evaluate.py is which model is being scored.
Both quantization and evaluation force CPU, since dynamic quantization
has no CUDA kernel.
"""

import json
import os

import torch
from torch.nn import CrossEntropyLoss

from src.data.dataloader import create_dataloaders
from src.quantize_utils import build_quantized_model

CHECKPOINT_PATH = "models/distilbert_imdb_baseline.pt"
RESULT_PATH = "results/int8_accuracy.json"


def evaluate(model, test_loader, device):
    model.eval()
    criterion = CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            predictions = torch.argmax(outputs.logits, dim=1)

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total
    return average_loss, accuracy


def main():
    device = torch.device("cpu")  # dynamic quantization: CPU only
    print(f"Device: {device} (forced -- dynamic quantization has no CUDA kernel)")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    print("\nCreating test dataloader...")
    _, test_loader = create_dataloaders(batch_size=16)
    print(f"Testing batches: {len(test_loader)}")

    print("\nBuilding INT8 quantized model...")
    model = build_quantized_model(CHECKPOINT_PATH)
    print("Quantized model ready.")

    print("\nEvaluating (this will be slower than GPU FP32 -- CPU inference)...")
    test_loss, test_accuracy = evaluate(model, test_loader, device)

    print("\nEvaluation results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Accuracy (%): {test_accuracy * 100:.2f}%")

    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)
    results = {
        "model": "distilbert-base-uncased",
        "variant": "int8_dynamic_quantization",
        "checkpoint": CHECKPOINT_PATH,
        "dataset": "stanfordnlp/imdb",
        "split": "test",
        "device": str(device),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_accuracy_percent": test_accuracy * 100,
    }

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {RESULT_PATH}")
    print("\nCompare this against results/baseline_accuracy.json (FP32) "
          "to compute the accuracy cost of quantization.")


if __name__ == "__main__":
    main()
