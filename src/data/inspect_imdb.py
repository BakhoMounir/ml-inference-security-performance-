from datasets import load_dataset


def main():
    dataset = load_dataset("stanfordnlp/imdb")

    print(dataset)
    print("\nTrain example:")
    print(dataset["train"][0])

    print("\nTest example:")
    print(dataset["test"][0])

    print("\nLabel names:")
    print(dataset["train"].features["label"])


if __name__ == "__main__":
    main()