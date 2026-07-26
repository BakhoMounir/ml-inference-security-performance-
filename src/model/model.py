import torch
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertConfig,
)


MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2


def build_model(
    model_name=MODEL_NAME,
    num_labels=NUM_LABELS,
    checkpoint_path=None,
    offline=False,
):

    if offline:

        if checkpoint_path is None:
            raise ValueError(
                "offline=True requires checkpoint_path"
            )

        config = DistilBertConfig(
            vocab_size=30522,
            max_position_embeddings=512,
            n_layers=6,
            n_heads=12,
            dim=768,
            hidden_dim=3072,
            num_labels=num_labels,
        )

        model = DistilBertForSequenceClassification(
            config
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        model.load_state_dict(
            checkpoint
        )

        return model


    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )

    return model


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    model = build_model()

    model.to(device)

    print(
        f"Model loaded: {MODEL_NAME}"
    )

    print(
        "Number of parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )


if __name__ == "__main__":
    main()
