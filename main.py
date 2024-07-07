# Import necessary libraries
import pandas as pd
import re
import nltk
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from gensim.models import CoherenceModel, LdaModel
from gensim import corpora
import matplotlib.pyplot as plt
import os
from keybert import KeyBERT

# Download NLTK data files
nltk.download('stopwords')
nltk.download('wordnet')

# Load Spacy model
nlp = spacy.load("en_core_web_sm")

# Load dataset with low_memory=False to handle mixed types
print("Loading dataset...")
df = pd.read_csv(r"C:/Users/Fabio/Desktop/consumer_complaints/consumer_complaints.csv", low_memory=False)

# Select relevant column
texts = df['consumer_complaint_narrative'].dropna().reset_index(drop=True)

# Data preprocessing function
def preprocess_text(text):
    # Remove special characters and numbers
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    # Remove redacted tokens (xx, xxxx)
    words = [word for word in words if len(word) > 2 and not re.match(r'^x+$', word)]
    
    # Lemmatization
    words = [token.lemma_ for token in nlp(' '.join(words))]
    
    return ' '.join(words)

print("Starting preprocessing... This might take a while.")
# Preprocess texts
clean_texts = texts.apply(preprocess_text)
print("Preprocessing completed.")

# Convert texts to list for KeyBERT
clean_texts_list = clean_texts.tolist()

# Initialize KeyBERT model
kw_model = KeyBERT()

# Extract keywords from texts
keywords = [kw_model.extract_keywords(text) for text in clean_texts_list]

# Function to calculate coherence score
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

# Create dictionary and corpus for Gensim LDA
print("Creating dictionary and corpus...")
dictionary = corpora.Dictionary([text.split() for text in clean_texts])
corpus = [dictionary.doc2bow(text.split()) for text in clean_texts]
print("Dictionary and corpus created.")

# Compute coherence scores to find the optimal number of topics
print("Calculating coherence scores...")
model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=[text.split() for text in clean_texts], start=2, limit=20, step=1)

# Plot coherence scores
limit = 20
start = 2
step = 1
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.legend(("coherence_values"), loc='best')
plt.savefig("coherence_scores.png")
plt.show()

# Select the model with the highest coherence score
optimal_model = model_list[coherence_values.index(max(coherence_values))]
print("Optimal model selected.")

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Save the LDA topics to a CSV file
topics = optimal_model.print_topics(num_words=10)
topics_df = pd.DataFrame(topics, columns=['Topic', 'Words'])
topics_df.to_csv(os.path.join(script_dir, "gensim_lda_topics.csv"), index=False)

# Function to plot the top words for each topic
def plot_gensim_top_words(lda_model, num_words, title, filename):
    fig, axes = plt.subplots(2, 5, figsize=(40, 20), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        top_words = lda_model.show_topic(i, num_words)
        words, weights = zip(*top_words)
        ax.barh(words, weights, color='blue')
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
