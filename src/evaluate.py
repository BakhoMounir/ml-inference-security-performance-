import json
import os

import torch
from torch.nn import CrossEntropyLoss

from src.data.dataloader import create_dataloaders
from src.model.model import build_model


CHECKPOINT_PATH = "models/distilbert_imdb_baseline.pt"
RESULT_PATH = "results/baseline_accuracy.json"


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

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = criterion(outputs.logits, labels)

            predictions = torch.argmax(
                outputs.logits,
                dim=1,
            )

            total_loss += loss.item() * labels.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("\nCreating test dataloader...")

    _, test_loader = create_dataloaders(
        batch_size=16
    )

    print(
        f"Testing batches: {len(test_loader)}"
    )

    print("\nBuilding model...")

    model = build_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(checkpoint)

    model.to(device)

    print(
        f"Loaded checkpoint: {CHECKPOINT_PATH}"
    )

    print("\nEvaluating...")

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        device,
    )

    print("\nEvaluation results:")
    print(
        f"Test Loss: {test_loss:.4f}"
    )
    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )
    print(
        f"Test Accuracy (%): {test_accuracy * 100:.2f}%"
    )

    os.makedirs(
        os.path.dirname(RESULT_PATH),
        exist_ok=True,
    )

    results = {
        "model": "distilbert-base-uncased",
        "checkpoint": CHECKPOINT_PATH,
        "dataset": "stanfordnlp/imdb",
        "split": "test",
        "device": str(device),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "test_accuracy_percent": test_accuracy * 100,
    }

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"\nSaved results to: {RESULT_PATH}"
    )


if __name__ == "__main__":
    main()