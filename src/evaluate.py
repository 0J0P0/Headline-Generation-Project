from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)

import torch
from configs.settings import MODEL_DIR, PROCESSED_DATA_DIR
from src.data_loader import HeadlineDataset, MAX_INPUT_LEN, MAX_TARGET_LEN
from src.model_loader import load_bart_model
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from bert_score import score as bert_score

BATCH_SIZE = 32


def evaluate():
    # tokenizer = BartTokenizer.from_pretrained(PROCESSED_DATA_DIR/"bart/")
    # model = BartForConditionalGeneration.from_pretrained(PROCESSED_DATA_DIR/"bart/")
    tokenizer, model = load_bart_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # dataset = HeadlineDataset(MODEL_DIR / "val.csv", tokenizer)
    dataset = HeadlineDataset(PROCESSED_DATA_DIR / "val.csv", tokenizer)
    summaries = dataset.data["input_text"].tolist()[:5]
    references = dataset.data["target_text"].tolist()[:5]

    model.eval()  # turn on eval mode

    predictions = []
    for summary in summaries:
        inputs = tokenizer(
            summary, return_tensors="pt", max_length=MAX_INPUT_LEN, truncation=True
        ).to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        output_ids = model.generate(input_ids=input_ids, max_length=MAX_TARGET_LEN)
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred)

    # METEOR
    total_score = 0

    # ROUGE (measures f1)
    total_rouge1 = 0  # unigram
    total_rougeL = 0  # Longest Common Subsequence
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Calculation
    for pred, refs in zip(predictions, references):
        refs_tokens = word_tokenize(refs.lower())
        pred_tokens = word_tokenize(pred.lower())
        total_score += meteor_score([pred_tokens], refs_tokens)  # METEOR

        scores = scorer.score(refs, pred)  # ROUGE
        total_rouge1 += scores["rouge1"].fmeasure
        total_rougeL += scores["rougeL"].fmeasure

    # Average calculation
    average_score = total_score / len(predictions)
    print("Average METEOR Score:", average_score)

    avg_rouge1 = total_rouge1 / len(predictions)
    avg_rougeL = total_rougeL / len(predictions)
    print(f"Average ROUGE-1 F1: {avg_rouge1:.4f}")
    print(f"Average ROUGE-L F1: {avg_rougeL:.4f}")

    P, R, F1 = bert_score(predictions, references, lang="en", verbose=True)  # BERT
    print(f"Average BERTScore Precision: {P.mean().item():.4f}")
    print(f"Average BERTScore Recall: {R.mean().item():.4f}")
    print(f"Average BERTScore F1: {F1.mean().item():.4f}")


if __name__ == "__main__":
    evaluate()
