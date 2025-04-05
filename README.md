
# Cross-Lingual Word Embedding Alignment (English-Hindi)

## Project Overview

This project aims to align pre-trained monolingual word embeddings from English and Hindi into a shared vector space, allowing for effective cross-lingual transfer of semantic information. This alignment enables applications such as cross-lingual word translation, document classification, and information retrieval.

## Approach

The implementation follows a supervised approach using the Procrustes method, which involves learning a linear transformation between the source (English) and target (Hindi) embedding spaces. The key steps in the approach are:

### 1. Data Preparation

- **Word Embeddings**: Used pre-trained FastText embeddings for both English and Hindi, limiting to the top 100,000 most frequent words in each language
- **Bilingual Lexicon**: Leveraged the MUSE dataset to extract word translation pairs for supervised alignment
- **Data Split**: Divided the bilingual lexicon into training (80%) and test (20%) sets

### 2. Embedding Alignment with Procrustes Method

- **Vector Preparation**: Extracted corresponding word vectors from both languages based on the training lexicon
- **Vector Normalization**: Normalized vectors to unit length to focus on direction rather than magnitude
- **Procrustes Analysis**: Applied SVD (Singular Value Decomposition) to learn an orthogonal transformation matrix that preserves angles and distances between word vectors
- **Matrix Application**: Used the learned transformation matrix to map English embeddings to the Hindi space

### 3. Evaluation

- **Word Translation Task**: Implemented a nearest-neighbor search to translate English words to Hindi
- **Metrics**:
  - Precision@1 (proportion of correct translations at rank 1): 0.1583
  - Precision@5 (proportion of correct translations in top 5 results): 0.3092
- **Similarity Analysis**: Analyzed the distribution of cosine similarities between aligned word pairs
- **Ablation Study**: Assessed the impact of bilingual lexicon size on alignment quality using different training dictionary sizes (5k, 10k, 20k word pairs)

### 4. Example Translations

Provided example translations for common English words like "computer", "language", "school", etc., with their corresponding Hindi translations and similarity scores.

## Key Components

The implementation includes the following key components:

1. **Data Loading**: Functions to download and load FastText embeddings and the MUSE bilingual lexicon
2. **Matrix Preparation**: Code to extract and prepare embedding matrices for alignment
3. **Procrustes Alignment**: Implementation of the Procrustes method using SVD
4. **Translation Function**: Function to translate words using the aligned embeddings
5. **Evaluation Metrics**: Code to compute Precision@1 and Precision@5
6. **Similarity Analysis**: Analysis of cosine similarities between aligned word pairs
7. **Ablation Study**: Evaluation with different dictionary sizes
8. **Visualization**: Plots for similarity distribution and ablation study results

## Results and Observations

- The model achieves moderate success in aligning embeddings between these two quite different languages
- The ablation study reveals that increasing the dictionary size generally improves alignment quality
- Analysis of cosine similarities shows a distribution centered around 0.6-0.7, indicating reasonable alignment
- Example translations demonstrate that the model can effectively capture semantic relationships across languages for many common words

## Future Improvements

Potential areas for enhancement include:
- Implementing CSLS (Cross-Domain Similarity Local Scaling) to address the hubness problem
- Exploring unsupervised methods for alignment to reduce dependency on bilingual lexicons
- Fine-tuning the alignment using domain-specific data for particular applications
- Incorporating subword information to handle morphologically rich languages better

## Implementation Details

The implementation is in Python using libraries such as NumPy, Pandas, Matplotlib, and scikit-learn. The code is structured to be modular and reusable for other language pairs.