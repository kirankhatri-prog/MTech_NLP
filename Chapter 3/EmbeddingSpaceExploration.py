"""
Assignment 2: Embedding Space Exploration
Uses real pretrained GloVe (50d, Wikipedia+Gigaword) vectors via gensim.

Install deps:  pip install gensim scikit-learn matplotlib numpy
Run:           python embedding_exploration.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gensim.downloader as api
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ------------------------------------------------------------------
# 1. Load pretrained embeddings (downloads once, then cached locally)
# ------------------------------------------------------------------
print("Loading GloVe embeddings (50-dim)...")
model = api.load('glove-wiki-gigaword-50')
print(f"Vocabulary size: {len(model.key_to_index)}")

# ------------------------------------------------------------------
# 2. Cosine similarity computations
# ------------------------------------------------------------------
pairs = [("king", "queen"), ("man", "woman"), ("cat", "dog"),
         ("king", "cabbage"), ("happy", "joyful"),
         ("paris", "france"), ("king", "man")]

print("\n--- Cosine similarities ---")
for a, b in pairs:
    print(f"cos({a}, {b}) = {model.similarity(a, b):.4f}")

# ------------------------------------------------------------------
# 3. Vector arithmetic / analogies
# ------------------------------------------------------------------
def analogy(pos, neg, topn=5):
    return model.most_similar(positive=pos, negative=neg, topn=topn)

print("\n--- king - man + woman ---")
for word, score in analogy(['king', 'woman'], ['man']):
    print(f"{word}: {score:.4f}")

print("\n--- paris - france + italy ---")
for word, score in analogy(['paris', 'italy'], ['france']):
    print(f"{word}: {score:.4f}")

print("\n--- walked - walk + swim ---")
for word, score in analogy(['walked', 'swim'], ['walk']):
    print(f"{word}: {score:.4f}")

# ------------------------------------------------------------------
# 4. PCA / t-SNE visualization of semantic clusters
# ------------------------------------------------------------------
words = ['king', 'queen', 'prince', 'princess', 'throne',
         'cat', 'dog', 'lion', 'tiger', 'elephant',
         'france', 'paris', 'italy', 'rome', 'spain', 'madrid',
         'apple', 'banana', 'orange', 'grape', 'mango',
         'happy', 'sad', 'angry', 'joyful', 'furious']

vectors = np.array([model[w] for w in words])

pca = PCA(n_components=2)
pca_result = pca.fit_transform(vectors)
print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")

tsne = TSNE(n_components=2, perplexity=8, random_state=42, init='pca')
tsne_result = tsne.fit_transform(vectors)

categories = (['royalty'] * 5 + ['animals'] * 5 + ['geo'] * 6 +
              ['fruits'] * 5 + ['emotions'] * 5)
color_map = {'royalty': '#e74c3c', 'animals': '#3498db', 'geo': '#2ecc71',
             'fruits': '#f39c12', 'emotions': '#9b59b6'}
colors = [color_map[c] for c in categories]

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, result, title in [(axes[0], pca_result, 'PCA'), (axes[1], tsne_result, 't-SNE')]:
    ax.scatter(result[:, 0], result[:, 1], c=colors, s=100)
    for i, w in enumerate(words):
        ax.annotate(w, (result[i, 0], result[i, 1]), fontsize=9,
                     xytext=(3, 3), textcoords='offset points')
    ax.set_title(f'{title} projection of GloVe-50d embeddings')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('embedding_clusters.png', dpi=150)
print("\nSaved visualization to embedding_clusters.png")
