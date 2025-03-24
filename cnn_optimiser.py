import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# ==================== 1. Define Text Preprocessing Functions ====================

def remove_html(text):
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    emoji_pattern = re.compile("["  # Emojis Unicode range
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()

# ==================== 2. Load & Preprocess Data ====================

df = pd.read_csv("tensorflow.csv")
df["text"] = df["Title"].astype(str) + " " + df["Body"].astype(str)
df = df.rename(columns={"class": "label"})
df["label"] = df["label"].astype(int)

# Apply preprocessing
df["text"] = df["text"].apply(remove_html)
df["text"] = df["text"].apply(remove_emoji)
df["text"] = df["text"].apply(clean_str)
df["tokens"] = df["text"].apply(lambda x: word_tokenize(str(x).lower()) if pd.notna(x) else [])

# ==================== 3. Load GloVe Word Embeddings ====================

def load_glove_embeddings(filepath, embedding_dim=100):
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings

glove_path = "glove.6B.100d.txt"
glove_embeddings = load_glove_embeddings(glove_path)

def text_to_embedding_indices(text, vocab, max_len=50):
    indices = [vocab[word] if word in vocab else np.zeros(100) for word in text]
    return indices[:max_len] + [np.zeros(100)] * (max_len - len(indices))

# ==================== 4. Define Dataset ====================

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==================== 5. Define the Optuna Objective Function ====================

# Store all results
trial_results = []

def objective(trial):
    """Objective function for Optuna hyperparameter tuning (runs each configuration 5 times)."""

    # **1️⃣ Sample Hyperparameters**
    learning_rate = trial.suggest_float('lr', 1e-5, 1, log=True)  # ✅ Fix for deprecated suggest_loguniform
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # ✅ Fix: Store kernel sizes as string & convert back to tuple
    kernel_sizes_options = ["(2,4,6)", "(3,5,7)", "(3,4,5)", "(5,7,9)"]
    kernel_sizes_str = trial.suggest_categorical('kernel_sizes', kernel_sizes_options)
    kernel_sizes = tuple(map(int, kernel_sizes_str.strip("()").split(",")))  # Convert back to tuple

    num_epochs = trial.suggest_int('epochs', 5, 100)
    activation_fn = trial.suggest_categorical('activation_fn', ['ReLU', 'LeakyReLU', 'GELU', 'SiLU'])

    # **2️⃣ Run each configuration 5 times and average the metrics**
    num_repeats = 5
    f1_scores, accuracies, precisions, recalls, auc_scores = [], [], [], [], []

    for _ in range(num_repeats):
        # **3️⃣ Data Preparation**
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df["tokens"].tolist(), df["label"].tolist(), test_size=0.2, random_state=np.random.randint(10000)
        )

        train_sequences = [text_to_embedding_indices(text, glove_embeddings) for text in train_texts]
        test_sequences = [text_to_embedding_indices(text, glove_embeddings) for text in test_texts]

        # ✅ Fix: Convert to NumPy first before converting to PyTorch tensor
        train_sequences_np = np.array(train_sequences, dtype=np.float32)
        train_labels_np = np.array(train_labels, dtype=np.int64)
        test_sequences_np = np.array(test_sequences, dtype=np.float32)
        test_labels_np = np.array(test_labels, dtype=np.int64)

        train_sequences = torch.tensor(train_sequences_np)
        train_labels = torch.tensor(train_labels_np)
        test_sequences = torch.tensor(test_sequences_np)
        test_labels = torch.tensor(test_labels_np)

        train_data = DataLoader(TextDataset(train_sequences, train_labels), batch_size=batch_size, shuffle=True)
        test_data = DataLoader(TextDataset(test_sequences, test_labels), batch_size=batch_size, shuffle=False)

        # **4️⃣ Define CNN Model**
        class CNNTextClassifier(nn.Module):
            def __init__(self, embedding_dim, num_classes):
                super(CNNTextClassifier, self).__init__()
                self.conv1 = nn.Conv1d(embedding_dim, 128, kernel_size=kernel_sizes[0], padding=1)
                self.conv2 = nn.Conv1d(128, 128, kernel_size=kernel_sizes[1], padding=1)
                self.conv3 = nn.Conv1d(128, 128, kernel_size=kernel_sizes[2], padding=1)
                self.batch_norm1 = nn.BatchNorm1d(128)
                self.batch_norm2 = nn.BatchNorm1d(128)
                self.batch_norm3 = nn.BatchNorm1d(128)
                self.activation = getattr(nn, activation_fn)()
                self.global_max_pool = nn.AdaptiveMaxPool1d(1)
                self.fc = nn.Linear(128, num_classes)

            def forward(self, x):
                x = x.permute(0, 2, 1)
                x = self.activation(self.batch_norm1(self.conv1(x)))
                x = self.activation(self.batch_norm2(self.conv2(x)))
                x = self.activation(self.batch_norm3(self.conv3(x)))
                x = self.global_max_pool(x).squeeze(-1)
                x = self.fc(x)
                return x

        # **5️⃣ Initialize Model**
        device = torch.device("cpu")
        model = CNNTextClassifier(embedding_dim=100, num_classes=2).to(device)

        # **6️⃣ Training**
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            model.train()
            for X_batch, y_batch in train_data:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch.long())
                loss.backward()
                optimizer.step()

        # **7️⃣ Evaluation**
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_data:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

        # Compute Metrics
        f1_scores.append(f1_score(all_labels, all_preds, average='macro'))
        accuracies.append(accuracy_score(all_labels, all_preds))
        precisions.append(precision_score(all_labels, all_preds, average='macro'))
        recalls.append(recall_score(all_labels, all_preds, average='macro'))
        auc_scores.append(roc_auc_score(all_labels, all_preds))

    # **8️⃣ Compute and Store Averages**
    trial_results.append({
        'Trial': trial.number,
        'Learning Rate': learning_rate,
        'Batch Size': batch_size,
        'Kernel Sizes': kernel_sizes,
        'Epochs': num_epochs,
        'Activation Function': activation_fn,
        'F1 Score': float(np.mean(f1_scores)),  # ✅ Fix np.float64 issue
        'Accuracy': float(np.mean(accuracies)),
        'Precision': float(np.mean(precisions)),
        'Recall': float(np.mean(recalls)),
        'AUC Score': float(np.mean(auc_scores))
    })

    # Print current trial results
    print(trial_results[-1])

    return trial_results[-1]['F1 Score']  # **F1 Score is the primary metric**


# ==================== 6. Run Optuna Optimization ====================

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Convert results to DataFrame & save to CSV
df_results = pd.DataFrame(trial_results)
df_results.to_csv("optuna_hyperparameter_results.csv", index=False)

print("Best Hyperparameters:", study.best_params)
