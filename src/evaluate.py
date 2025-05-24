from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    PegasusTokenizer,
    PegasusForConditionalGeneration,
)

from configs.settings import MODEL_DIR, PROCESSED_DATA_DIR
from src.data_loader import HeadlineDataset, MAX_INPUT_LEN, MAX_TARGET_LEN
from nltk.translate import meteor
from rouge_score import rouge
from bert_score import bert_score


def evaluate():
    model = BartForConditionalGeneration.from_pretrained(MODEL_DIR / "/bart/")
    tokenizer = BartTokenizer.from_pretrained(MODEL_DIR / "/bart/")

    dataset = HeadlineDataset(PROCESSED_DATA_DIR / "val.csv", tokenizer)
    summaries = dataset.data["input_text"].tolist()
    references = dataset.data["target_text"].tolist()

    model.eval()  # turn on eval mode

    predictions = []
    for summary in summaries:
        inputs = tokenizer(
            summary, return_tensors="pt", truncation=True, max_length=MAX_INPUT_LEN
        )
        output_ids = model.generate(**inputs, max_length=MAX_TARGET_LEN)
        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        predictions.append(pred)

    # METEOR
    total_score = 0

    # ROUGE (measures f1)
    total_rouge1 = 0  # unigram
    total_rougeL = 0  # Longest Common Subsequence
    scorer = rouge.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    # Calculation
    for pred, refs in zip(predictions, references):
        total_score += meteor(refs, pred)  # METEOR

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
