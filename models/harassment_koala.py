from typing import List
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

harass_model_name = "KoalaAI/Text-Moderation"
device = "cuda" if torch.cuda.is_available() else "cpu"

harass_model = AutoModelForSequenceClassification.from_pretrained(harass_model_name).to(device)
harass_tokenizer = AutoTokenizer.from_pretrained(harass_model_name)


def harassment_score(history: List[str]) -> float:
    """Return harassment severity score between 0 and 1, given full dialogue history."""
    text = "\n".join(history)

    if not text or text.strip() == "":
        return 0.0

    inputs = harass_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        outputs = harass_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze()

    id2label = harass_model.config.id2label
    return float({id2label[i]: probs[i] for i in range(len(probs))}.get("S", 0.0))
