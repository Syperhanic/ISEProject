import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re
import nltk
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download necessary resources
nltk.download("punkt")
nltk.download("stopwords")

# ==================== 1. Define Text Preprocessing Functions ====================

def remove_html(text):
    """Remove HTML tags using a regex."""
    html = re.compile(r'<.*?>')
    return html.sub(r'', text)

def remove_emoji(text):
    """Remove emojis using a regex pattern."""
    emoji_pattern = re.compile("["  # Emojis Unicode range
                               u"\U0001F600-\U0001F64F"
                               u"\U0001F300-\U0001F5FF"
                               u"\U0001F680-\U0001F6FF"
                               u"\U0001F1E0-\U0001F1FF"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_stopwords(text):
    """Remove stopwords from the text.
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word.lower() not in stop_words])"""
    return text

def clean_str(text):
    """Clean text by removing special characters and extra spaces."""
    text = re.sub(r"[^A-Za-z0-9(),.!?\'\`]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)  # Replace multiple spaces with a single space
    return text.strip().lower()

# ==================== 2. Load & Preprocess Data ====================

# Load the dataset
df = pd.read_csv("caffe.csv")

# Merge "Title" and "Body" into a new "text" column
df["text"] = df["Title"].astype(str) + " " + df["Body"].astype(str)

# Rename class column
df = df.rename(columns={"class": "label"})

# Convert labels to integers (ensure 0 and 1)
df["label"] = df["label"].astype(int)

# Apply text preprocessing
df["text"] = df["text"].apply(remove_html)
df["text"] = df["text"].apply(remove_emoji)
df["text"] = df["text"].apply(remove_stopwords)
df["text"] = df["text"].apply(clean_str)

# Tokenize the text
df["tokens"] = df["text"].apply(lambda x: word_tokenize(str(x).lower()) if pd.notna(x) else [])

# Split dataset (80% train, 20% test)
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["tokens"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42
)

# ==================== 3. Manually Load GloVe Word Embeddings ====================

def load_glove_embeddings(filepath, embedding_dim=100):
    """Load GloVe embeddings from a file."""
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings

# Load manually downloaded GloVe file
glove_path = "glove.6B.100d.txt"
glove_embeddings = load_glove_embeddings(glove_path)

# Convert words to embedding indices
def text_to_embedding_indices(text, vocab, max_len=50):
    indices = [vocab[word] if word in vocab else np.zeros(100) for word in text]
    return indices[:max_len] + [np.zeros(100)] * (max_len - len(indices))

train_sequences = [text_to_embedding_indices(text, glove_embeddings) for text in train_texts]
test_sequences = [text_to_embedding_indices(text, glove_embeddings) for text in test_texts]

# Convert to PyTorch tensors
train_sequences, train_labels = torch.tensor(train_sequences, dtype=torch.float32), torch.tensor(train_labels)
test_sequences, test_labels = torch.tensor(test_sequences, dtype=torch.float32), torch.tensor(test_labels)

# ==================== 4. Create Dataset & DataLoader ====================

class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create DataLoader
train_data = DataLoader(TextDataset(train_sequences, train_labels), batch_size=64, shuffle=True)
test_data = DataLoader(TextDataset(test_sequences, test_labels), batch_size=64, shuffle=False)

# ==================== 5. Define CNN Model (with Batch Normalization) ====================

class CNNTextClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CNNTextClassifier, self).__init__()

        # ✅ Reverting back to only 3 convolutional layers
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=4, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=1)

        # ✅ Keeping batch normalization
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.batch_norm3 = nn.BatchNorm1d(128)

        # Activation function
        self.activation = nn.LeakyReLU()
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        # Fully connected layer
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.activation(self.batch_norm1(self.conv1(x)))
        x = self.activation(self.batch_norm2(self.conv2(x)))
        x = self.activation(self.batch_norm3(self.conv3(x)))

        x = self.global_max_pool(x).squeeze(-1)
        x = self.fc(x)

        return x  # Return raw logits


# Initialize Model
embedding_dim = 100
num_classes = 2

device = torch.device("cpu")
model = CNNTextClassifier(embedding_dim, num_classes).to(device)

# ==================== 6. Train the CNN Model ====================

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.004288702295008052)

num_epochs = 45

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_data:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.squeeze().long())

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# ==================== 7. Evaluate the Model ====================

# Evaluate the model once on the fixed train-test split
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_data:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model(X_batch)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

# Compute and print final results
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
auc_score = roc_auc_score(all_labels, all_preds)

print("\n=== CNN Classification Results ===")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"AUC Score: {auc_score:.4f}")

print(f"{accuracy:.4f}\t{precision:.4f}\t{recall:.4f}\t{f1:.4f}\t{auc_score:.4f}")

