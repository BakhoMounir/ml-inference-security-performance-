import torch
from transformers import DistilBertForSequenceClassification


MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2


def build_model(
    model_name=MODEL_NAME,
    num_labels=NUM_LABELS,
):
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

    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )


if __name__ == "__main__":
    main()