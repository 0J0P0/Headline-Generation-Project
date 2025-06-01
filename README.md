# 🎬 Headline Generation Project

This project aims to generate accurate and creative **movie titles** from detailed **movie plot summaries** using **transformer-based models** like BART and PEGASUS.

It includes:

* Data preprocessing
* Fine-tuning pretrained models
* Generation and evaluation of movie titles
* Support for both BART and PEGASUS models

---

## 📁 Project Structure

```
Headline-Generation-Project/
│
├── data/
│   ├── raw/             # Raw dataset files
│   ├── processed/       # Processed dataset files
│   │   ├── train.csv
│   │   └── test.csv
│
├── src/
│   ├── train.py         # Script for training the model
│   ├── generate.py      # Script for generating titles
│   └── evaluate.py      # Script for evaluating generated titles
│
├── models/
│   ├── bart/            # Folder to store BART fine-tuned model (safe tensors)
│   └── pegasus/         # Folder to store PEGASUS fine-tuned model (safe tensors)
│
├── scritps/
│   ├── train.sh         # Shell script to execute training
│   └── eval.sh          # Shell script to evaluate the model
│
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore file
└── README.md
```

---

## ⚙️ Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/0J0P0/Headline-Generation-Project.git
   cd Headline-Generation-Project
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Training a Model

You can fine-tune either **BART** or **PEGASUS** on our movie plot data.

### 📌 Option 1: Using Python directly

```bash
# Train BART
python -m src.train bart

# Train PEGASUS
python -m src.train pegasus
```

> 🧠 After training, the `.safetensors` file is saved inside `bart/` or `pegasus/` folders.

### 📌 Option 2: Using Shell Script

```bash
bash scripts/train.sh
```

---

## 📝 Generating Titles

Once the model is trained, generate movie titles from test data:

```bash
python -m src.generate bart
python -m src.generate pegasus
```

---

## 📊 Evaluating the Results

Evaluate model performance using **ROUGE**, **METEOR**, and **BERTScore**.

```bash
python -m src.evaluate bart
python -m src.evaluate pegasus
```

Or:

```bash
bash eval.sh
```

---

## 📈 Evaluation Metrics

* **ROUGE-1 / ROUGE-L**
* **METEOR**
* **BERTScore**

Evaluation is based on F1 scores between the generated and true movie titles.

---

## 📚 Dataset Format

The CSV files should have the following format:

```csv
input_text;target_text
"Full movie plot goes here...";"Expected Movie Title"
```
