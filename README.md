  # Consumer Complaints Analysis

This project analyzes consumer complaints using Natural Language Processing (NLP) techniques to extract prevalent topics and keywords. The analysis helps identify common issues and provides insights for decision-makers.

## Overview
The goal of this project is to preprocess consumer complaints data, vectorize the text using Bag of Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF), extract keywords using KeyBERT, and perform topic modeling using Latent Semantic Analysis (LSA) and Latent Dirichlet Allocation (LDA). The coherence score is used to determine the optimal number of topics for LDA.

## Requirements
To run this project, you need the following Python libraries:
+ pandas
+ re
+ nltk
+ spacy
+ sklearn
+ gensim
+ matplotlib
+ seaborn
+ keybert
+ scipy

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/fabseren/consumer_complaints.git
   cd consumer_complaints
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   source venv/bin/activate      # On macOS/Linux
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
4. Download the consumer_complaints.csv dataset and place it in the project directory.

## Usage
To run the analysis, execute the following command:
  ```bash
  python main.py
  ```
## Script Overview
+ Preprocessing: Cleans the text data by removing special characters, numbers, stopwords, and performing lemmatization.
+ Vectorization: Converts text data into numerical vectors using BoW and TF-IDF methods.
+ Keyword Extraction: Uses KeyBERT to extract top keywords from a sampled percentage of the data.
+ Comparison: Compares the top features from BoW and TF-IDF.
+ Topic Modeling: Uses LSA with TF-IDF and LDA with the optimal number of topics determined by coherence score.
+ Plotting: Plots the top words for each topic using Seaborn.

## Results
The results of the analysis are saved in the following files:
+ keybert_keywords.csv: Extracted keywords using KeyBERT.
+ bow_vectors.csv: BoW vectorized data.
+ tfidf_vectors.npz: TF-IDF vectorized data (saved in sparse format).
+ comparison_bow_tfidf.png: Visual comparison of BoW and TF-IDF top features.
+ lsa_topics.csv: Topics extracted using LSA.
+ lsa_topics.png: Plot of LSA topics.
+ gensim_lda_topics.csv: Topics extracted using Gensim LDA.
+ gensim_lda_topics.png: Plot of Gensim LDA topics.
+ coherence_scores.png: Plot of coherence scores to determine the optimal number of topics.

## Note
Due to the massive amount of data contained in the consumer_complaints.csv file, the keyBERT keywords extraction and the coherence score are calculated just on a random 1% sample of the dataset. To adjust the percentage of the utilised sample according to your available computation power, modify line 56 for keyBERT extraction and line 175 for the coherence score calculation.
