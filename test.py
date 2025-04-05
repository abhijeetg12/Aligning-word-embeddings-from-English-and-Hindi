# Cross-Lingual Word Embedding Alignment for English and Hindi
# Author: Claude
# Date: April 4, 2025

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import requests
import zipfile
import gzip
import fasttext
import fasttext.util
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from tqdm.notebook import tqdm
from google.colab import drive

# Mount Google Drive (uncomment if using Colab)
# drive.mount('/content/drive')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

#------------------------------------------------------------------------------
# 1. Data Preparation
#------------------------------------------------------------------------------

def download_file(url, filename):
    """Download a file from URL if it doesn't exist"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"{filename} already exists")

def load_vectors(fname, nmax=100000):
    """Load word vectors from text file"""
    print(f"Loading vectors from {fname}...")
    word_vectors = {}
    
    with open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore') as fin:
        n, d = map(int, fin.readline().split())
        count = 0
        
        for line in fin:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            vector = np.array(tokens[1:], dtype=float)
            word_vectors[word] = vector
            count += 1
            
            if count == nmax:
                break
    
    print(f"Loaded {len(word_vectors)} word vectors with dimension {d}")
    return word_vectors

def download_fasttext_embeddings():
    """Download pre-trained FastText embeddings for English and Hindi"""
    # Download English embeddings
    en_emb_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.vec"
    en_emb_file = "wiki.en.vec"
    download_file(en_emb_url, en_emb_file)
    
    # Download Hindi embeddings
    hi_emb_url = "https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.hi.vec"
    hi_emb_file = "wiki.hi.vec"
    download_file(hi_emb_url, hi_emb_file)
    
    return en_emb_file, hi_emb_file

def download_muse_dataset():
    """Download MUSE dataset for supervised alignment"""
    # Download MUSE dataset
    muse_url = "https://dl.fbaipublicfiles.com/arrival/dictionaries/en-hi.txt"
    muse_file = "en-hi.txt"
    download_file(muse_url, muse_file)
    
    return muse_file

def load_muse_dictionary(file_path, n_max=None):
    """Load word translation pairs from MUSE dataset"""
    print(f"Loading dictionary from {file_path}...")
    word_pairs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n_max is not None and i >= n_max:
                break
            src, tgt = line.rstrip().split()
            word_pairs.append((src, tgt))
    
    print(f"Loaded {len(word_pairs)} word pairs")
    return word_pairs

def split_dictionary(dictionary, train_ratio=0.8):
    """Split dictionary into training and test sets"""
    np.random.shuffle(dictionary)
    split_idx = int(len(dictionary) * train_ratio)
    train_dict = dictionary[:split_idx]
    test_dict = dictionary[split_idx:]
    return train_dict, test_dict

def prepare_embedding_matrices(src_vectors, tgt_vectors, train_dict):
    """Prepare source and target embedding matrices for Procrustes alignment"""
    # Get words that exist in our dictionaries
    src_words = [pair[0] for pair in train_dict if pair[0] in src_vectors]
    tgt_words = [pair[1] for pair in train_dict if pair[0] in src_vectors and pair[1] in tgt_vectors]
    
    # Filter out pairs where either word is not in our embeddings
    valid_pairs = [(pair[0], pair[1]) for pair in train_dict 
                   if pair[0] in src_vectors and pair[1] in tgt_vectors]
    
    if len(valid_pairs) == 0:
        raise ValueError("No valid translation pairs found in the embeddings.")
    
    src_words = [pair[0] for pair in valid_pairs]
    tgt_words = [pair[1] for pair in valid_pairs]
    
    # Extract vectors for these words
    X = np.vstack([src_vectors[word] for word in src_words])
    Y = np.vstack([tgt_vectors[word] for word in tgt_words])
    
    print(f"Prepared matrices with {X.shape[0]} word pairs")
    return X, Y, src_words, tgt_words, valid_pairs

#------------------------------------------------------------------------------
# 2. Embedding Alignment using Procrustes Method
#------------------------------------------------------------------------------

def procrustes_align(X, Y):
    """
    Learn alignment matrix using Procrustes method
    
    Args:
        X: Source embedding matrix (n_samples, dim)
        Y: Target embedding matrix (n_samples, dim)
    
    Returns:
        W: Orthogonal transformation matrix
    """
    # Compute X^T * Y
    XTY = X.T.dot(Y)
    
    # SVD decomposition
    U, _, Vt = np.linalg.svd(XTY)
    
    # Orthogonal transformation matrix
    W = U.dot(Vt)
    
    return W

def normalize_embeddings(emb):
    """Normalize embeddings to unit length"""
    norms = np.sqrt(np.sum(emb**2, axis=1, keepdims=True))
    return emb / norms

#------------------------------------------------------------------------------
# 3. Evaluation
#------------------------------------------------------------------------------

def translate_words(src_word, src_vectors, tgt_vectors, W, k=5):
    """
    Translate a source word to the target language
    
    Args:
        src_word: Source word to translate
        src_vectors: Dictionary of source word vectors
        tgt_vectors: Dictionary of target word vectors
        W: Transformation matrix
        k: Number of translations to return
    
    Returns:
        List of top-k translations with scores
    """
    if src_word not in src_vectors:
        return []
    
    # Get source word vector and apply transformation
    src_vec = src_vectors[src_word]
    transformed_vec = src_vec.dot(W)
    
    # Normalize the transformed vector
    transformed_vec = transformed_vec / np.linalg.norm(transformed_vec)
    
    # Compute cosine similarities with all target words
    similarities = []
    for tgt_word, tgt_vec in tgt_vectors.items():
        # Normalize target vector
        tgt_vec_norm = tgt_vec / np.linalg.norm(tgt_vec)
        sim = np.dot(transformed_vec, tgt_vec_norm)
        similarities.append((tgt_word, sim))
    
    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def evaluate_translation(test_dict, src_vectors, tgt_vectors, W, k=5):
    """
    Evaluate word translation accuracy
    
    Args:
        test_dict: List of (source_word, target_word) pairs
        src_vectors: Dictionary of source word vectors
        tgt_vectors: Dictionary of target word vectors
        W: Transformation matrix
        k: Number of translations to consider (for Precision@k)
    
    Returns:
        Precision@1 and Precision@k scores
    """
    correct_1 = 0
    correct_k = 0
    total = 0
    
    for src_word, tgt_word in test_dict:
        if src_word in src_vectors and tgt_word in tgt_vectors:
            translations = translate_words(src_word, src_vectors, tgt_vectors, W, k)
            
            if len(translations) > 0:
                total += 1
                
                # Check Precision@1
                if translations[0][0] == tgt_word:
                    correct_1 += 1
                
                # Check Precision@k
                if any(trans[0] == tgt_word for trans in translations):
                    correct_k += 1
    
    p1 = correct_1 / total if total > 0 else 0
    pk = correct_k / total if total > 0 else 0
    
    return p1, pk

def analyze_semantic_similarity(test_dict, src_vectors, tgt_vectors, W):
    """
    Analyze cross-lingual semantic similarity between word pairs
    
    Args:
        test_dict: List of (source_word, target_word) pairs
        src_vectors: Dictionary of source word vectors
        tgt_vectors: Dictionary of target word vectors
        W: Transformation matrix
    
    Returns:
        DataFrame with word pairs and their cosine similarities
    """
    results = []
    
    for src_word, tgt_word in test_dict:
        if src_word in src_vectors and tgt_word in tgt_vectors:
            # Get vectors
            src_vec = src_vectors[src_word]
            tgt_vec = tgt_vectors[tgt_word]
            
            # Apply transformation to source vector
            transformed_vec = src_vec.dot(W)
            
            # Normalize vectors
            transformed_vec = transformed_vec / np.linalg.norm(transformed_vec)
            tgt_vec = tgt_vec / np.linalg.norm(tgt_vec)
            
            # Compute cosine similarity
            sim = np.dot(transformed_vec, tgt_vec)
            
            results.append({
                'source_word': src_word,
                'target_word': tgt_word,
                'similarity': sim
            })
    
    return pd.DataFrame(results)

def ablation_study(dictionary, src_vectors, tgt_vectors, sizes=[5000, 10000, 20000]):
    """
    Conduct ablation study to assess the impact of dictionary size
    
    Args:
        dictionary: Full list of (source_word, target_word) pairs
        src_vectors: Dictionary of source word vectors
        tgt_vectors: Dictionary of target word vectors
        sizes: List of dictionary sizes to test
    
    Returns:
        DataFrame with results for each dictionary size
    """
    results = []
    
    for size in sizes:
        if size > len(dictionary):
            size = len(dictionary)
        
        print(f"\nTesting with dictionary size: {size}")
        
        # Get subset of dictionary
        train_dict = dictionary[:size]
        test_dict = dictionary[size:size+1000]  # Use next 1000 pairs for testing
        
        # Prepare matrices
        X, Y, _, _, _ = prepare_embedding_matrices(src_vectors, tgt_vectors, train_dict)
        
        # Normalize embeddings
        X = normalize_embeddings(X)
        Y = normalize_embeddings(Y)
        
        # Align embeddings
        W = procrustes_align(X, Y)
        
        # Evaluate
        p1, p5 = evaluate_translation(test_dict, src_vectors, tgt_vectors, W, k=5)
        
        results.append({
            'dictionary_size': size,
            'precision@1': p1,
            'precision@5': p5
        })
    
    return pd.DataFrame(results)

def visualize_ablation_results(results):
    """
    Visualize results of the ablation study
    
    Args:
        results: DataFrame with ablation study results
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(results['dictionary_size'], results['precision@1'], marker='o', label='Precision@1')
    plt.plot(results['dictionary_size'], results['precision@5'], marker='s', label='Precision@5')
    
    plt.title('Impact of Dictionary Size on Alignment Quality')
    plt.xlabel('Dictionary Size')
    plt.ylabel('Precision')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ablation_results.png')
    plt.show()

def visualize_embeddings_tsne(src_words, tgt_words, src_vectors, tgt_vectors, W, n_sample=100):
    """
    Visualize aligned embeddings using t-SNE
    
    Args:
        src_words: List of source words
        tgt_words: List of target words
        src_vectors: Dictionary of source word vectors
        tgt_vectors: Dictionary of target word vectors
        W: Transformation matrix
        n_sample: Number of words to sample for visualization
    """
    from sklearn.manifold import TSNE
    
    # Sample words
    if len(src_words) > n_sample:
        indices = np.random.choice(len(src_words), n_sample, replace=False)
        src_words_sample = [src_words[i] for i in indices]
        tgt_words_sample = [tgt_words[i] for i in indices]
    else:
        src_words_sample = src_words
        tgt_words_sample = tgt_words
    
    # Get embeddings
    src_embs = np.array([src_vectors[w] for w in src_words_sample])
    tgt_embs = np.array([tgt_vectors[w] for w in tgt_words_sample])
    
    # Transform source embeddings
    src_embs_transformed = src_embs.dot(W)
    
    # Normalize
    src_embs_transformed = normalize_embeddings(src_embs_transformed)
    tgt_embs = normalize_embeddings(tgt_embs)
    
    # Combine embeddings
    combined_embs = np.vstack([src_embs_transformed, tgt_embs])
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_embs)-1))
    embeddings_2d = tsne.fit_transform(combined_embs)
    
    # Split back
    src_embs_2d = embeddings_2d[:len(src_words_sample)]
    tgt_embs_2d = embeddings_2d[len(src_words_sample):]
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    plt.scatter(src_embs_2d[:, 0], src_embs_2d[:, 1], c='blue', label='English', alpha=0.7)
    plt.scatter(tgt_embs_2d[:, 0], tgt_embs_2d[:, 1], c='red', label='Hindi', alpha=0.7)
    
    for i, (src_word, tgt_word) in enumerate(zip(src_words_sample, tgt_words_sample)):
        # Draw line connecting translation pairs
        plt.plot([src_embs_2d[i, 0], tgt_embs_2d[i, 0]], 
                 [src_embs_2d[i, 1], tgt_embs_2d[i, 1]], 'k-', alpha=0.1)
        
        # Annotate words (uncomment if needed)
        # plt.annotate(src_word, src_embs_2d[i], fontsize=8, alpha=0.8)
        # plt.annotate(tgt_word, tgt_embs_2d[i], fontsize=8, alpha=0.8)
    
    plt.title('t-SNE Visualization of Aligned Word Embeddings')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('tsne_visualization.png')
    plt.show()

#------------------------------------------------------------------------------
# 4. Main Execution
#------------------------------------------------------------------------------

def main():
    # 1. Data Preparation
    print("Step 1: Data Preparation")
    
    # Download pre-trained FastText embeddings
    en_emb_file, hi_emb_file = download_fasttext_embeddings()
    
    # Load word vectors (limiting to top 100,000 most frequent words)
    en_vectors = load_vectors(en_emb_file)
    hi_vectors = load_vectors(hi_emb_file)
    
    # Download MUSE dataset
    muse_file = download_muse_dataset()
    
    # Load bilingual lexicon
    dictionary = load_muse_dictionary(muse_file)
    
    # Split dictionary into train and test sets
    train_dict, test_dict = split_dictionary(dictionary)
    
    print(f"Training dictionary size: {len(train_dict)}")
    print(f"Test dictionary size: {len(test_dict)}")
    
    # 2. Embedding Alignment
    print("\nStep 2: Embedding Alignment")
    
    # Prepare matrices for Procrustes alignment
    X, Y, src_words, tgt_words, valid_pairs = prepare_embedding_matrices(en_vectors, hi_vectors, train_dict)
    
    # Normalize embeddings
    X = normalize_embeddings(X)
    Y = normalize_embeddings(Y)
    
    # Learn alignment matrix
    W = procrustes_align(X, Y)
    
    # 3. Evaluation
    print("\nStep 3: Evaluation")
    
    # Evaluate translation accuracy
    p1, p5 = evaluate_translation(test_dict, en_vectors, hi_vectors, W, k=5)
    print(f"Precision@1: {p1:.4f}")
    print(f"Precision@5: {p5:.4f}")
    
    # Analyze semantic similarity
    similarity_df = analyze_semantic_similarity(test_dict[:100], en_vectors, hi_vectors, W)
    print("\nSemantic Similarity Analysis (sample):")
    print(similarity_df.head())
    
    # Save similarity analysis
    similarity_df.to_csv('semantic_similarity_analysis.csv', index=False)
    
    # Visualize aligned embeddings
    visualize_embeddings_tsne(src_words, tgt_words, en_vectors, hi_vectors, W)
    
    # Conduct ablation study
    print("\nConducting Ablation Study...")
    ablation_results = ablation_study(dictionary, en_vectors, hi_vectors, 
                                      sizes=[1000, 5000, 10000, 20000])
    
    print("\nAblation Study Results:")
    print(ablation_results)
    
    # Save ablation results
    ablation_results.to_csv('ablation_results.csv', index=False)
    
    # Visualize ablation results
    visualize_ablation_results(ablation_results)
    
    # 4. Example translations
    print("\nExample Translations:")
    example_words = ["hello", "world", "computer", "language", "science"]
    
    for word in example_words:
        if word in en_vectors:
            translations = translate_words(word, en_vectors, hi_vectors, W, k=3)
            print(f"\n{word} -> {[t[0] for t in translations]}")
            print(f"Similarity scores: {[f'{t[1]:.4f}' for t in translations]}")

if __name__ == "__main__":
    main()