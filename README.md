# Fine-Tuning DistilBERT on IMDB Sentiment Analysis

## 📌 Project Overview
This project demonstrates how to fine-tune **DistilBERT**, a lightweight transformer-based language model, for **binary sentiment classification** using the IMDB movie reviews dataset. It highlights the practical power of **transfer learning** in NLP.

## 🎯 Objectives
- Fine-tune DistilBERT on a subset of the IMDB dataset (1,000 reviews).
- Classify reviews as either **positive** or **negative**.
- Evaluate model performance using accuracy, F1 score, and confusion matrix.
- Visualize training dynamics and results.

## 🧠 Theoretical Background
- **Transformers** use self-attention to understand context across the entire input.
- **DistilBERT** is a distilled version of BERT, offering similar performance with fewer parameters and faster inference.
- **Transfer learning** allows us to leverage pretrained models for downstream tasks with limited data.

## 🧰 Tools and Environment
- **Platform:** Google Colab
- **Libraries:** `transformers`, `datasets`, `torch`, `scikit-learn`, `matplotlib`, `seaborn`, `tensorboard`
- **Storage:** Google Drive (for logs and model checkpoints)

## 🗃 Dataset and Preprocessing
- **Dataset:** IMDB (50K labeled reviews; used 1K for prototyping)
- **Tokenization:** `DistilBertTokenizerFast` with padding and truncation to 256 tokens
- **Preprocessing:** Lowercasing, whitespace stripping

## 🧪 Model and Training
- **Model:** `DistilBertForSequenceClassification` with 2 output labels
- **Training Config:**
  - Epochs: 2
  - Batch Size: 32
  - Learning Rate: 5e-5
  - Optimizer: AdamW
- **Trainer API:** Used from Hugging Face to simplify the training loop

## 📊 Results
- **Accuracy & F1 Score:** Above 0.80 on small dataset
- **Confusion Matrix:** Strong diagonal → good separation
- **Error Analysis:** Sarcasm and ambiguous phrasing were common misclassification causes

## 📈 Visualizations
- Confusion Matrix
- Training vs Validation Loss
- Accuracy & F1 Score curves

## 🎥 Demo
Optionally includes a live or notebook-based demo:
- Input: Movie review
- Output: Predicted sentiment
- Includes: classification report, loss curves, sample predictions

## 🧩 Lessons Learned
- Transfer learning is powerful even with small datasets
- Preprocessing significantly impacts performance
- Hugging Face tools simplify complex NLP workflows
- Small datasets can’t generalize well to production

## 🔮 Next Steps
- Train on full 50K IMDB dataset
- Test with larger models (BERT, RoBERTa)
- Add early stopping and data augmentation
- Hyperparameter tuning using Optuna or grid search
- Use MLflow or Weights & Biases for experiment tracking


## ✍️ Author
Shreya Thakur
002818444

---


