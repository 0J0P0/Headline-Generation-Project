from transformers import BartTokenizer, BartForConditionalGeneration
import torch

from configs.settings import MODEL_DIR


def generate_headline(text, path):
    """
    Generate a headline for the given text using a pre-trained BART model.
    Args:
        text (str): The input text for which to generate a headline.
        Returns:
        str: The generated headline.
    """
    tokenizer = BartTokenizer.from_pretrained(path)
    model = BartForConditionalGeneration.from_pretrained(path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    model.eval()
    
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=32,
            num_beams=4,
            early_stopping=True,
        )
    
    headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return headline
if __name__ =="__main__":
    text= "The quick brown fox jumps over the lazy dog. This is a test sentence to generate a headline."
    title = generate_headline(text)
    print(f"Generated Movie Title: {title}")