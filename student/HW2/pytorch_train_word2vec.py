import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

import numpy as np

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, centers, contexts):
        self.centers = centers
        self.contexts = contexts

    def __len__(self):
        return self.centers.shape[0]

    def __getitem__(self, idx):
        return self.centers[idx], self.contexts[idx]


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)
        nn.init.uniform_(self.in_embed.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center_ids, context_ids):
        v = self.in_embed(center_ids)
        u = self.out_embed(context_ids)
        if u.dim() == 2:
            return (v * u).sum(dim=1)
        return (v.unsqueeze(1) * u).sum(dim=2)

    def get_embeddings(self):
        return self.in_embed.weight


# Load processed data
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

pairs_obj = None
for k in ["skip_gram_pairs", "skipgram_pairs", "pairs", "data", "train_pairs", "skipgram_df"]:
    if isinstance(data, dict) and k in data:
        pairs_obj = data[k]
        break
if pairs_obj is None:
    pairs_obj = data

try:
    centers_np = np.array(pairs_obj["center"], dtype=np.int64)
    contexts_np = np.array(pairs_obj["context"], dtype=np.int64)
except Exception:
    pairs_arr = np.array(pairs_obj, dtype=np.int64)
    centers_np = pairs_arr[:, 0]
    contexts_np = pairs_arr[:, 1]

centers = torch.as_tensor(centers_np, dtype=torch.long)
contexts = torch.as_tensor(contexts_np, dtype=torch.long)

word2idx = data["word2idx"] if isinstance(data, dict) and "word2idx" in data else None
idx2word = data["idx2word"] if isinstance(data, dict) and "idx2word" in data else None
vocab_size = len(word2idx) if word2idx is not None else int(max(int(centers.max()), int(contexts.max())) + 1)


# Precompute negative sampling distribution below
counts = torch.bincount(contexts, minlength=vocab_size).float()
neg_dist = counts.pow(0.75)
neg_dist = neg_dist / neg_dist.sum()


# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# Dataset and DataLoader
dataset = SkipGramDataset(centers, contexts)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=0)


# Model, Loss, Optimizer
model = Word2Vec(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM).to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def make_targets(center, context, vocab_size):
    return context

# Training loop
neg_dist_device = neg_dist.to(device)

for epoch in range(EPOCHS):
    total_loss = 0.0
    for center_ids, context_ids in loader:
        center_ids = center_ids.to(device)
        context_ids = context_ids.to(device)

        optimizer.zero_grad()

        pos_logits = model(center_ids, context_ids)
        pos_loss = loss_fn(pos_logits, torch.ones_like(pos_logits))

        neg_context_ids = torch.multinomial(
            neg_dist_device,
            num_samples=center_ids.shape[0] * NEGATIVE_SAMPLES,
            replacement=True,
        ).view(center_ids.shape[0], NEGATIVE_SAMPLES)

        pos = context_ids.view(-1, 1)
        mask = neg_context_ids.eq(pos)
        while mask.any():
            n_bad = int(mask.sum().item())
            resample = torch.multinomial(neg_dist_device, num_samples=n_bad, replacement=True)
            neg_context_ids[mask] = resample
            mask = neg_context_ids.eq(pos)

        neg_logits = model(center_ids, neg_context_ids)
        neg_loss = loss_fn(neg_logits, torch.zeros_like(neg_logits))

        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())

    avg_loss = total_loss / len(loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} | avg loss: {avg_loss:.4f}")


# Save embeddings and mappings
# embeddings = model.get_embeddings()
embeddings = model.get_embeddings().detach().cpu().numpy()
if word2idx is None:
    word2idx = {str(i): i for i in range(vocab_size)}
if idx2word is None:
    idx2word = {i: w for w, i in word2idx.items()}
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': word2idx, 'idx2word': idx2word}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
