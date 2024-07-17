# NLP Poetry Language Modeling

## Overview
This project aims to build language models for poetry using natural language processing (NLP) techniques, focusing on renowned poets Robert Frost and Edgar Allan Poe. Various approaches from bi-gram models to LSTM-based models are explored for generating and analyzing poetry texts.

## Dataset
The primary text sources used are:
- Robert Frost's Poetry
- Edgar Allan Poe's Poetry

## Tasks and Implementations
1. **Bi-gram Markov Language Model**
   - Implemented using matrix representation to capture word transition probabilities.
   - Preprocessed text, handled sentence boundaries, and applied Laplace smoothing.

2. **Sentence Probability and Smoothing**
   - Computed sentence probabilities using bi-gram model.
   - Enhanced with Laplace smoothing for improved performance on unseen sequences.

3. **Perplexity Calculation**
   - Developed a function to compute perplexity based on the smoothed bi-gram model.

4. **Markov Classifier**
   - Built separate bi-gram language models for Frost and Poe.
   - Evaluated classification accuracy and created a confusion matrix.

5. **PPMI Word Co-occurrence and Embeddings**
   - Created a word-word co-occurrence matrix with PPMI.
   - Derived word embeddings using Truncated SVD and visualized them in 2D.

6. **LSTM Language Model**
   - Formulated an LSTM-based model trained on sequences of words from both poets.
   - Initialized with SVD embeddings and trained to predict the next word in sequences.

7. **Text Generation and Evaluation**
   - Generated poetry using bi-gram and LSTM models.
   - Compared effectiveness in terms of text quality and coherence.

8. **Bonus Tasks**
   - Implemented a 2nd order Markov model and enhanced text generation techniques.

## Results and Discussion
- Discussed model performance metrics and insights into generated text quality.
- Addressed challenges encountered during implementation.

## Conclusion
- Summarized project objectives, achievements, and potential improvements.
- Emphasized the significance of NLP techniques in analyzing and generating poetic texts.
