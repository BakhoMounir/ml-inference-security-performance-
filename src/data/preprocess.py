from datasets import load_from_disk
from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


def load_imdb_dataset():
    dataset = load_from_disk("imdb_dataset")

    return {
        "train": dataset["train"],
        "test": dataset["test"],
    }


def tokenize_dataset(dataset, tokenizer):
    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
        )

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
    )

    tokenized_dataset.set_format("torch")

    return tokenized_dataset


def load_and_tokenize():
    dataset = load_imdb_dataset()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    tokenized_dataset = {
        split: tokenize_dataset(
            dataset[split],
            tokenizer,
        )
        for split in ["train", "test"]
    }

    return tokenized_dataset


def main():

    tokenized_dataset = load_and_tokenize()

    print("Splits:", tokenized_dataset.keys())

    for split in ["train", "test"]:

        print(f"\n{split}:")
        print(tokenized_dataset[split])
        print("Columns:")
        print(tokenized_dataset[split].column_names)

    print("\nFirst tokenized example:")
    print(tokenized_dataset["train"][0])

    print("\nInput IDs shape:")
    print(
        tokenized_dataset["train"][0]["input_ids"].shape
    )

    print("\nAttention mask shape:")
    print(
        tokenized_dataset["train"][0]["attention_mask"].shape
    )

    print("\nLabel:")
    print(
        tokenized_dataset["train"][0]["label"]
    )


if __name__ == "__main__":
    main()
