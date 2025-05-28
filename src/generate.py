from transformers import (BartTokenizer, BartForConditionalGeneration)
import torch

from configs.settings import MODEL_DIR

def generate_headline(text):
    tokenizer=BartTokenizer.from_pretrained(MODEL_DIR/"bart/")
    model=BartForConditionalGeneration.from_pretrained(MODEL_DIR/"bart/")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    inputs=tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs={k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs=model.generate(
            input_ids=inputs["input_ids"],
            max_length=32,
            num_beams=4, 
            early_stopping=True
        )
    headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return headline
