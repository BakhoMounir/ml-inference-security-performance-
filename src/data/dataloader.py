from torch.utils.data import DataLoader

from .preprocess import load_and_tokenize


BATCH_SIZE = 16


def create_dataloaders(
    batch_size=BATCH_SIZE,
):
    tokenized_dataset = load_and_tokenize()

    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        tokenized_dataset["test"],
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader


def main():

    train_loader, test_loader = create_dataloaders(
        batch_size=16
    )

    train_batch = next(iter(train_loader))

    print("Train batch keys:")
    print(train_batch.keys())

    print(
        "Train input IDs shape:",
        train_batch["input_ids"].shape,
    )

    print(
        "Train attention mask shape:",
        train_batch["attention_mask"].shape,
    )

    print(
        "Train labels shape:",
        train_batch["label"].shape,
    )

    print(
        "\nTrain batches:",
        len(train_loader),
    )

    print(
        "Test batches:",
        len(test_loader),
    )


if __name__ == "__main__":
    main()