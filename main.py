# Import libraries
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora
import matplotlib.pyplot as plt
import seaborn as sns
import os
from keybert import KeyBERT
import random
from scipy import sparse

def main():
    # Download NLTK data files
    nltk.download('stopwords')
    nltk.download('wordnet')

    # Load Spacy model
    nlp = spacy.load("en_core_web_sm")

    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv(r"C:/Users/Fabio/Desktop/consumer_complaints/consumer_complaints.csv", low_memory=False)

    # Select relevant column
    texts = df['consumer_complaint_narrative'].dropna().reset_index(drop=True)

    # Data preprocessing function
    def preprocess_text(text):
        # Remove special characters and numbers, convert to lowercase, remove stopwords and redacted tokens. Lemmatitazion.
        text = re.sub(r'\W', ' ', text)
        text = re.sub(r'\d', ' ', text)
        text = text.lower()
        stop_words = set(stopwords.words('english'))
        words = text.split()
        words = [word for word in words if word not in stop_words]
        words = [word for word in words if len(word) > 2 and not re.match(r'^x+$', word)]
        words = [token.lemma_ for token in nlp(' '.join(words))]
        
        return ' '.join(words)

    print("Starting preprocessing...")
    # Preprocess texts
    clean_texts = texts.apply(preprocess_text)
    print("Preprocessing completed.")

    # Convert texts to list for KeyBERT
    clean_texts_list = clean_texts.tolist()

    # Set the percentage of the dataset to sample for KeyBERT (e.g., 0.10 for 10%)
    sample_percentage = 0.01
    sample_size = int(len(clean_texts_list) * sample_percentage)

    # Sample the data for KeyBERT
    sampled_texts = random.sample(clean_texts_list, sample_size)
    kw_model = KeyBERT()

    # Extract top 5 keywords from the sampled texts
    print(f"Extracting keywords using KeyBERT")
    keywords = [kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5) for text in sampled_texts]

    # Convert keywords to DataFrame and save to CSV
    keywords_df = pd.DataFrame(keywords)
    keywords_df.to_csv("keybert_keywords.csv", index=False)
    print("Keywords extracted and saved to keybert_keywords.csv.")

    # Bag of Words
    print("Vectorizing using Bag of Words...")
    count_vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=5000)
    count_vectors = count_vectorizer.fit_transform(clean_texts)
    print("Bag of Words vectorization completed.")

    # TF-IDF
    print("Vectorizing using TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=5000)
    tfidf_vectors = tfidf_vectorizer.fit_transform(clean_texts)
    print("TF-IDF vectorization completed.")

    # Save the TF-IDF matrix to a file
    sparse.save_npz("tfidf_vectors.npz", tfidf_vectors)

    # Compare BoW and TF-IDF
    print("Comparing BoW and TF-IDF...")
    bow_sums = count_vectors.sum(axis=0).A1
    tfidf_sums = tfidf_vectors.sum(axis=0).A1

    feature_names_bow = count_vectorizer.get_feature_names_out()
    feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()

    # Get top 10 features by frequency for BoW
    top_bow_indices = bow_sums.argsort()[-10:][::-1]
    top_bow_features = feature_names_bow[top_bow_indices]
    top_bow_values = bow_sums[top_bow_indices]

    # Get top 10 features by frequency for TF-IDF
    top_tfidf_indices = tfidf_sums.argsort()[-10:][::-1]
    top_tfidf_features = feature_names_tfidf[top_tfidf_indices]
    top_tfidf_values = tfidf_sums[top_tfidf_indices]

    # Visual comparison of the top 10 features
    plt.figure(figsize=(10, 5))
    plt.barh(top_bow_features, top_bow_values, color='blue', alpha=0.6, label='BoW')
    plt.barh(top_tfidf_features, top_tfidf_values, color='red', alpha=0.6, label='TF-IDF')
    plt.xlabel('Frequency')
    plt.title('Top 10 Features: BoW vs TF-IDF')
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison_bow_tfidf.png")
    plt.show()

    # LSA with TF-IDF
    print("Performing LSA...")
    lsa_model = TruncatedSVD(n_components=10, random_state=42)
    lsa_topic_matrix = lsa_model.fit_transform(tfidf_vectors)

    # Get the words associated with each topic
    terms = tfidf_vectorizer.get_feature_names_out()
    lsa_topics = {}
    for idx, component in enumerate(lsa_model.components_):
        terms_in_topic = [terms[i] for i in component.argsort()[:-11:-1]]
        lsa_topics[f"Topic {idx}"] = terms_in_topic

    lsa_topics_df = pd.DataFrame(lsa_topics)
    lsa_topics_df.to_csv("lsa_topics.csv", index=False)

    def plot_lsa_topics(lsa_model, terms, num_words=10):
        fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
        axes = axes.flatten()
        for idx, ax in enumerate(axes):
            # Get the sorted indices in descending order of weights
            sorted_indices = lsa_model.components_[idx].argsort()[::-1]
            # Select the top num_words terms and their corresponding weights
            top_terms_indices = sorted_indices[:num_words]
            top_terms = [terms[i] for i in top_terms_indices]
            top_weights = lsa_model.components_[idx][top_terms_indices]
            ax.barh(top_terms, top_weights, color='blue')
            ax.set_title(f"Topic {idx+1}")
            ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig("lsa_topics.png")
        plt.show()

    plot_lsa_topics(lsa_model, terms)
    print("LSA plotting completed.")

    # Function to calculate coherence score
    def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3, sample_size=1000):
        coherence_values = []
        model_list = []
        
        # Sample the corpus and texts
        sampled_corpus = corpus[:sample_size]
        sampled_texts = texts[:sample_size]

        for num_topics in range(start, limit, step):
            model = LdaModel(corpus=sampled_corpus, num_topics=num_topics, id2word=dictionary)
            model_list.append(model)
            coherencemodel = CoherenceModel(model=model, texts=sampled_texts, dictionary=dictionary, coherence='c_v')
            coherence_values.append(coherencemodel.get_coherence())
        return model_list, coherence_values

    # Create dictionary and corpus for Gensim LDA
    print("Creating dictionary and corpus...")
    dictionary = corpora.Dictionary([text.split() for text in clean_texts])
    corpus = [dictionary.doc2bow(text.split()) for text in clean_texts]
    print("Dictionary and corpus created.")

    # Compute coherence scores to find the optimal number of topics
    print("Calculating coherence scores...")
    model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=[text.split() for text in clean_texts], start=2, limit=20, step=1, sample_size=int(len(clean_texts) * 0.01))

    # Plot coherence scores
    limit = 20
    start = 2
    step = 1
    x = range(start, limit, step)
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=x, y=coherence_values, marker='o')
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.legend(["Coherence Score"], loc='best')
    plt.savefig("coherence_scores.png")
    plt.show()

    # Select the model with the highest coherence score
    optimal_model = model_list[coherence_values.index(max(coherence_values))]
    print("Optimal model selected with number of topics: ", optimal_model.num_topics)

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Save the LDA topics to a CSV file
    topics = optimal_model.print_topics(num_words=10)
    topics_df = pd.DataFrame(topics, columns=['Topic', 'Words'])
    topics_df.to_csv(os.path.join(script_dir, "gensim_lda_topics.csv"), index=False)


        # Function to plot the top words for each topic using Seaborn
    def plot_gensim_top_words(lda_model, num_words, title, filename):
        fig, axes = plt.subplots(2, 5, figsize=(40, 20), sharex=True)
        axes = axes.flatten()
        for i, ax in enumerate(axes):
            if i < lda_model.num_topics:  # Only plot if the index is within the number of topics
                top_words = lda_model.show_topic(i, num_words)
                words, weights = zip(*top_words)
                sns.barplot(x=list(weights), y=list(words), ax=ax, palette="viridis")
                ax.set_title(f'Topic {i + 1}', fontsize=18)
                ax.invert_yaxis()
                ax.tick_params(axis='both', which='major', labelsize=14)
        plt.suptitle(title, fontsize=24)
        plt.subplots_adjust(top=0.85, hspace=0.5, wspace=0.4)
        plt.savefig(os.path.join(script_dir, filename))
        plt.show()

    # Plot and save the Gensim LDA topics
    plot_gensim_top_words(optimal_model, 10, 'Top words per Gensim LDA topic', 'gensim_lda_topics.png')
    print("Plotting completed.")

if __name__ == "__main__":
    main()