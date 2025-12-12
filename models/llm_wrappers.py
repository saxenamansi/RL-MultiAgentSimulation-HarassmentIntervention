import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
device = "cuda"
dtype = torch.bfloat16


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="cuda",
    )
    return model, tokenizer


MODEL, TOKENIZER = load_model()


def generate_text(prompt: str,
                  max_new_tokens: int = 80,
                  temperature: float = 0.8,
                  top_p: float = 0.95) -> str:
    messages = [{"role": "user", "content": prompt}]

    input_ids = TOKENIZER.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        output_ids = MODEL.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=TOKENIZER.eos_token_id,
            eos_token_id=TOKENIZER.eos_token_id,
        )

    generated_ids = output_ids[0, input_ids.shape[-1]:]
    text = TOKENIZER.decode(generated_ids, skip_special_tokens=True)
    text = text.strip()
    if "." in text:
        text = text.rsplit(".", 1)[0] + "."

    return text.strip()


def llm_harasser(prompt: str) -> str:
    return generate_text(prompt, max_new_tokens=96, temperature=0.7, top_p=0.95)


def llm_victim(prompt: str) -> str:
    return generate_text(prompt, max_new_tokens=96, temperature=0.7, top_p=0.95)


def llm_intervener(prompt: str) -> str:
    return generate_text(prompt, max_new_tokens=64, temperature=0.4, top_p=0.9)
