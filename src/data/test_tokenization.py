from transformers import AutoTokenizer


MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    texts = [
        "This movie was excellent.",
        "This is a much longer review. " * 100,
    ]

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    for i, text in enumerate(texts):
        attention_mask = encoded["attention_mask"][i]

        real_tokens = int(attention_mask.sum())
        padding_tokens = int((attention_mask == 0).sum())

        print(f"\nExample {i + 1}")
        print("Original text characters:", len(text))
        print("Input shape:", encoded["input_ids"][i].shape)
        print("Real tokens:", real_tokens)
        print("Padding tokens:", padding_tokens)
        print("Total positions:", len(encoded["input_ids"][i]))

    print("\nBatch input_ids shape:", encoded["input_ids"].shape)
    print("Batch attention_mask shape:", encoded["attention_mask"].shape)


if __name__ == "__main__":
    main()