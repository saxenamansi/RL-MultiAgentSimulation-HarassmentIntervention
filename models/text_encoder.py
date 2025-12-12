import numpy as np
from sentence_transformers import SentenceTransformer

ENCODER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_encoder = SentenceTransformer(ENCODER_MODEL_NAME, device="cuda")


def text_encoder(text: str) -> np.ndarray:
    emb = _encoder.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.astype(np.float32)
