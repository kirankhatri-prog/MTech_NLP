"""
Assignment 3: Build a Word2Vec Model (Skip-Gram with Negative Sampling)
Implemented from scratch in PyTorch on a small custom corpus.
"""
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------------------------------------------
# 1. Small toy corpus (you can replace this with any text file)
# ---------------------------------------------------------------
corpus = """
the king ruled the great kingdom with wisdom and courage
the queen supported the king and cared for the people
the prince and princess played in the royal garden
a dog chased the cat around the house
the cat climbed a tree to escape the dog
dogs and cats are common household pets
the man walked to the market to buy fruit
the woman walked to the market to buy vegetables
apples and bananas are sweet fruits
the king and queen visited the market together
the prince learned to ride a horse
the princess learned to paint beautiful pictures
the dog barked loudly at the stranger
the cat purred softly on the warm mat
the man and woman worked together in the field
the kingdom celebrated the new prince's birth
the queen wore a golden crown studded with jewels
the king wore a crown of silver and gold
horses and dogs are loyal animals
the market sold apples bananas oranges and grapes
"""

# ---------------------------------------------------------------
# 2. Preprocessing: tokenize + build vocabulary
# ---------------------------------------------------------------
def tokenize(text):
    text = text.lower()
    tokens = re.findall(r"[a-z']+", text)
    return tokens

tokens = tokenize(corpus)
word_counts = Counter(tokens)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(vocab)
token_ids = [word2idx[t] for t in tokens]

print(f"Corpus tokens: {len(tokens)}")
print(f"Vocabulary size: {vocab_size}")

# ---------------------------------------------------------------
# 3. Generate skip-gram training pairs (center, context)
# ---------------------------------------------------------------
WINDOW_SIZE = 2

def generate_pairs(token_ids, window_size):
    pairs = []
    for i, center in enumerate(token_ids):
        window = random.randint(1, window_size)
        start = max(0, i - window)
        end = min(len(token_ids), i + window + 1)
        for j in range(start, end):
            if j != i:
                pairs.append((center, token_ids[j]))
    return pairs

pairs = generate_pairs(token_ids, WINDOW_SIZE)
print(f"Training pairs (center, context): {len(pairs)}")

# ---------------------------------------------------------------
# 4. Negative sampling distribution (unigram^0.75, as in the
#    original Word2Vec paper)
# ---------------------------------------------------------------
freqs = np.array([word_counts[idx2word[i]] for i in range(vocab_size)], dtype=np.float64)
neg_dist = freqs ** 0.75
neg_dist = neg_dist / neg_dist.sum()

def sample_negatives(batch_size, num_neg, exclude_idx=None):
    return np.random.choice(vocab_size, size=(batch_size, num_neg), p=neg_dist)

# ---------------------------------------------------------------
# 5. Skip-Gram model with negative sampling
# ---------------------------------------------------------------
class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embed_dim)   # center word embeddings
        self.out_embed = nn.Embedding(vocab_size, embed_dim)  # context word embeddings
        nn.init.uniform_(self.in_embed.weight, -0.5 / embed_dim, 0.5 / embed_dim)
        nn.init.uniform_(self.out_embed.weight, -0.5 / embed_dim, 0.5 / embed_dim)

    def forward(self, center, context, negatives):
        v_c = self.in_embed(center)              # (B, D)
        u_o = self.out_embed(context)             # (B, D)
        u_neg = self.out_embed(negatives)         # (B, K, D)

        # Positive score: log sigmoid(v_c . u_o)
        pos_score = torch.sum(v_c * u_o, dim=1)              # (B,)
        pos_loss = torch.nn.functional.logsigmoid(pos_score)

        # Negative score: log sigmoid(-v_c . u_neg)
        neg_score = torch.bmm(u_neg, v_c.unsqueeze(2)).squeeze(2)  # (B, K)
        neg_loss = torch.nn.functional.logsigmoid(-neg_score).sum(1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

# ---------------------------------------------------------------
# 6. Training loop
# ---------------------------------------------------------------
EMBED_DIM = 50
NUM_NEG = 5
BATCH_SIZE = 64
EPOCHS = 200
LR = 0.01

device = torch.device('cpu')
model = SkipGramNS(vocab_size, EMBED_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LR)

centers = torch.tensor([p[0] for p in pairs], dtype=torch.long)
contexts = torch.tensor([p[1] for p in pairs], dtype=torch.long)
n_samples = len(pairs)

for epoch in range(1, EPOCHS + 1):
    perm = torch.randperm(n_samples)
    total_loss = 0.0
    n_batches = 0
    for start in range(0, n_samples, BATCH_SIZE):
        idx = perm[start:start + BATCH_SIZE]
        batch_center = centers[idx].to(device)
        batch_context = contexts[idx].to(device)
        neg = torch.tensor(sample_negatives(len(idx), NUM_NEG), dtype=torch.long).to(device)

        optimizer.zero_grad()
        loss = model(batch_center, batch_context, neg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if epoch % 20 == 0 or epoch == 1:
        print(f"Epoch {epoch:3d} | avg loss: {total_loss / n_batches:.4f}")

# ---------------------------------------------------------------
# 7. Evaluation: nearest neighbours + simple analogy
# ---------------------------------------------------------------
embeddings = model.in_embed.weight.detach().numpy()

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

def most_similar(word, topn=5):
    if word not in word2idx:
        return []
    vec = embeddings[word2idx[word]]
    sims = []
    for w, i in word2idx.items():
        if w == word:
            continue
        sims.append((w, cosine_sim(vec, embeddings[i])))
    sims.sort(key=lambda x: -x[1])
    return sims[:topn]

print("\n--- Nearest neighbours (trained on tiny corpus) ---")
for w in ['king', 'dog', 'market', 'queen']:
    print(f"\nMost similar to '{w}':")
    for neighbour, score in most_similar(w):
        print(f"  {neighbour:12s} {score:.4f}")

# Save embeddings + vocab for later inspection
np.save('embeddings.npy', embeddings)
with open('vocab.txt', 'w') as f:
    for w in vocab:
        f.write(w + '\n')

print("\nTraining complete. Embeddings and vocab saved.")
