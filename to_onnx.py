import numpy as np
from transformers import AutoTokenizer
from onnxruntime import InferenceSession


def onnx_inference(
        text: str,
        session: InferenceSession,
        tokenizer: AutoTokenizer,
        max_length: int
) -> np.ndarray:
    """Инференс модели с помощью ONNX Runtime.

    @param text: входной текст для классификации
    @param session: ONNX Runtime-сессия
    @param tokenizer: токенизатор
    @param max_length: максимальная длина последовательности в токенах
    @return: логиты на выходе из модели
    """
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="np",
    )
    input_feed = {
        "input_ids": inputs["input_ids"].astype(np.int64)
    }
    outputs = session.run(
        output_names=["output"],
        input_feed=input_feed
    )[0]
    return outputs

outputs = onnx_inference(
        text='как дела?',
        session: InferenceSession,
        tokenizer: AutoTokenizer,
        max_length=200)