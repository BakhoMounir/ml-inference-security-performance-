import os
import time

import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from src.data.dataloader import create_dataloaders
from src.model.model import build_model
from src.utils.reproducibility import set_seed


MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 5e-5

MODEL_SAVE_PATH = "models/distilbert_imdb_baseline.pt"


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    loss_fn,
    device,
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    for batch_idx, batch in enumerate(dataloader):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits

        loss = loss_fn(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

        if batch_idx % 100 == 0:

            print(
                f"Batch {batch_idx}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f}"
            )

    epoch_time = (
        time.time() - start_time
    )

    average_loss = (
        total_loss / len(dataloader)
    )

    accuracy = correct / total

    return (
        average_loss,
        accuracy,
        epoch_time,
    )


def evaluate(
    model,
    dataloader,
    loss_fn,
    device,
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for batch in dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs.logits

            loss = loss_fn(
                logits,
                labels,
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / len(dataloader)
    )

    accuracy = correct / total

    return (
        average_loss,
        accuracy,
    )


def main():

    set_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        "\nCreating dataloaders..."
    )

    train_loader, test_loader = (
        create_dataloaders(
            batch_size=BATCH_SIZE,
        )
    )

    print(
        f"Training batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Testing batches: "
        f"{len(test_loader)}"
    )

    print(
        "\nBuilding model..."
    )

    model = build_model(
        model_name=MODEL_NAME,
        num_labels=2,
    )

    model.to(device)

    loss_fn = CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print(
        "\nStarting training...\n"
    )

    for epoch in range(EPOCHS):

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )

        train_loss, train_accuracy, epoch_time = (
            train_one_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=device,
            )
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            dataloader=test_loader,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            "\nEpoch results:"
        )

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train Accuracy: "
            f"{train_accuracy:.4f}"
        )

        print(
            f"Test Loss: "
            f"{test_loss:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{test_accuracy:.4f}"
        )

        print(
            f"Epoch Time: "
            f"{epoch_time:.2f} seconds"
        )

    print(
        "\nSaving model..."
    )

    os.makedirs(
        os.path.dirname(MODEL_SAVE_PATH),
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        MODEL_SAVE_PATH,
    )

    print(
        f"Model saved to: "
        f"{MODEL_SAVE_PATH}"
    )

    print(
        "\nTraining completed."
    )


if __name__ == "__main__":
    main()