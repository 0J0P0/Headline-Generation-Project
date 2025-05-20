from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)


def load_bart_model(model_name="facebook/bart-base"):
    """
    Load a BART model and tokenizer from the Hugging Face model hub.
    Args:
        model_name (str): The name of the BART model to load. Default is 'facebook/bart-base'.
    Returns:
        tokenizer: The tokenizer for the BART model.
        model: The BART model.
    """
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model


def load_pegasus_model(model_name="google/pegasus-xsum"):
    """
    Load a Pegasus model and tokenizer from the Hugging Face model hub.
    Args:
        model_name (str): The name of the Pegasus model to load. Default is 'google/pegasus-xsum'.
    Returns:
        tokenizer: The tokenizer for the Pegasus model.
        model: The Pegasus model.
    """
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    return tokenizer, model
