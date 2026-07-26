"""
Dynamic INT8 quantization for the DistilBERT classifier.

PyTorch dynamic quantization only supports CPU execution (fbgemm/qnnpack
backends have no CUDA kernels) -- so this always returns a CPU model.
Only nn.Linear layers are quantized (the dominant cost in a Transformer),
leaving embeddings and LayerNorm in FP32, which is standard practice.
"""

import torch

from src.model.model import build_model


def build_quantized_model(checkpoint_path: str):
    model = build_model()
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()

    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    return quantized_model
