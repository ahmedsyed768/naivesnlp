import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Function to load dataset
@st.cache
def load_data(filename):
    data = pd.read_csv(filename)
    return data

# Preprocessing function
def preprocess_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [token.lower() for token in tokens]
    # Remove punctuation and non-alphanumeric characters
    tokens = [token for token in tokens if token.isalnum()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

# Function for sentiment analysis
def perform_sentiment_analysis(data):
    # Preprocess text data
    data['preprocessed_feedback'] = data['feedback'].apply(preprocess_text)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(data['preprocessed_feedback'], data['sentiment'], test_size=0.2, random_state=42)

    # Feature extraction using Word2Vec
    tokenized_text = X_train.apply(lambda x: x.split())
    w2v_model = Word2Vec(tokenized_text, size=100, window=5, min_count=1, workers=4)
    X_train_vectorized = w2v_model.wv[X_train.tolist()]
    X_test_vectorized = w2v_model.wv[X_test.tolist()]

    # Initialize and train the Naive Bayes classifier
    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train_vectorized, y_train)

    # Predict sentiment on the testing set
    predictions = naive_bayes.predict(X_test_vectorized)

    # Evaluate the model
    accuracy = accuracy_score(y_test, predictions)
    st.write("Accuracy:", accuracy)
    st.write("\nClassification Report:")
    st.write(classification_report(y_test, predictions))
    
    # Calculate precision, recall, and F1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
    st.write("Precision:", precision)
    st.write("Recall:", recall)
    st.write("F1 Score:", f1_score)
    
    # Plot precision-recall curve
    probs = naive_bayes.predict_proba(X_test_vectorized)
    precision, recall, _ = precision_recall_curve(y_test, probs[:, 1])
    plt.plot(recall, precision, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    st.pyplot(plt)

# Main function
def main():
    st.title("Student Feedback Sentiment Analysis")
    st.sidebar.title("Upload Dataset")

    # Allow user to upload dataset
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.write("Preview of Uploaded Data:")
        st.write(data.head())

        st.write("Performing Sentiment Analysis...")
        perform_sentiment_analysis(data)

# Entry point
if __name__ == "__main__":
    main()
