import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import gensim.downloader as api
import copy
import matplotlib.pyplot as plt


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_split_data(seed=42):
    dataset = load_dataset("financial_phrasebank", "sentences_50agree", trust_remote_code=True)
    data = dataset["train"]

    texts = data["sentence"]
    labels = data["label"]

    train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
        texts, labels, test_size=0.15, random_state=seed, stratify=labels
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_val_texts,
        train_val_labels,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=train_val_labels,
    )

    return train_texts, val_texts, test_texts, train_labels, val_labels, test_labels


def load_fasttext_model():
    return api.load("fasttext-wiki-news-subwords-300")


def get_sentence_embedding(sentence, fasttext_model, embed_dim=300):
    tokens = sentence.split()
    vectors = []

    for token in tokens:
        if token in fasttext_model.key_to_index:
            vectors.append(fasttext_model.get_vector(token))

    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(embed_dim)


def create_embeddings(texts, fasttext_model):
    return np.array([get_sentence_embedding(t, fasttext_model) for t in texts], dtype=np.float32)


class SentimentDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=300, hidden_dim=256, num_classes=3, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.network(x)


def compute_class_weights(labels):
    unique_classes = np.unique(labels)
    class_weights = compute_class_weight("balanced", classes=unique_classes, y=labels)
    return torch.tensor(class_weights, dtype=torch.float32)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for embeddings, labels in dataloader:
        embeddings, labels = embeddings.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for embeddings, labels in dataloader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    return avg_loss, accuracy, macro_f1, all_preds, all_labels


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50):
    best_val_f1 = 0.0
    best_model_state = None

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_f1s = []
    val_f1s = []

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        train_loss_eval, train_acc, train_f1, _, _ = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        scheduler.step(val_f1)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train Macro F1: {train_f1:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  New best model with Val Macro F1: {best_val_f1:.4f}")

    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_f1s': train_f1s,
        'val_f1s': val_f1s
    }

    return best_model_state, best_val_f1, metrics


def plot_training_curves(metrics, save_path='outputs/mlp_training_curves.png'):
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(epochs, metrics['train_losses'], label='Train Loss', marker='o', markersize=3)
    axes[0].plot(epochs, metrics['val_losses'], label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss vs Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, metrics['train_accs'], label='Train Accuracy', marker='o', markersize=3)
    axes[1].plot(epochs, metrics['val_accs'], label='Val Accuracy', marker='s', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy vs Epochs')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, metrics['train_f1s'], label='Train Macro F1', marker='o', markersize=3)
    axes[2].plot(epochs, metrics['val_f1s'], label='Val Macro F1', marker='s', markersize=3)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Macro F1')
    axes[2].set_title('Macro F1 vs Epochs')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path='outputs/mlp_confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Negative', 'Neutral', 'Positive'],
           yticklabels=['Negative', 'Neutral', 'Positive'],
           title='Confusion Matrix - MLP Classifier',
           ylabel='True Label',
           xlabel='Predicted Label')
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def main():
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading and splitting data...")
    train_texts, val_texts, test_texts, train_labels, val_labels, test_labels = load_and_split_data(seed=42)

    print("Loading FastText model...")
    fasttext_model = load_fasttext_model()

    print("Creating embeddings...")
    train_embeddings = create_embeddings(train_texts, fasttext_model)
    val_embeddings = create_embeddings(val_texts, fasttext_model)
    test_embeddings = create_embeddings(test_texts, fasttext_model)

    print("Computing class weights...")
    class_weights = compute_class_weights(train_labels).to(device)

    print("Creating datasets and dataloaders...")
    train_dataset = SentimentDataset(train_embeddings, train_labels)
    val_dataset = SentimentDataset(val_embeddings, val_labels)
    test_dataset = SentimentDataset(test_embeddings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("Initializing model...")
    model = MLPClassifier(input_dim=300, hidden_dim=256, num_classes=3, dropout=0.3).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    print("Training model...")
    best_model_state, best_val_f1, metrics = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=50
    )

    print(f"\nBest Validation Macro F1: {best_val_f1:.4f}")

    print("Plotting training curves...")
    plot_training_curves(metrics, save_path='outputs/mlp_training_curves.png')

    print("Saving best model...")
    torch.save(best_model_state, "best_sentiment_mlp_model.pth")

    print("Evaluating on test set...")
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_f1, test_preds, test_labels_list = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Test Macro F1: {test_f1:.4f}")

    print("Plotting confusion matrix...")
    plot_confusion_matrix(test_labels_list, test_preds, save_path='outputs/mlp_confusion_matrix.png')


if __name__ == "__main__":
    main()