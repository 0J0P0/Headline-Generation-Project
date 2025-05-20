import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm

from src.model_loader import load_bart_model
from src.data_loader import HeadlineDataset
from configs.settings import MODEL_DIR, PROCESSED_DATA_DIR

BATCH_SIZE = 32
EPOCHS = 4
LEARNING_RATE = 5e-5


def train():
    """
    Train the BART model for headline generation.
    """
    # Load tokenizer and model
    tokenizer, model = load_bart_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Dataset and Dataloader
    dataset = HeadlineDataset(PROCESSED_DATA_DIR / "train.csv", tokenizer)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loop.set_postfix(loss=loss.item())

    model.save_pretrained(MODEL_DIR / "/bart/")
    tokenizer.save_pretrained(MODEL_DIR / "/bart/")


if __name__ == "__main__":
    train()
