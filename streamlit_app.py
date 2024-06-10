import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import io

# Downloading necessary NLTK data
nltk.download('vader_lexicon')


class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def preprocess_text(self, text):
        if isinstance(text, str):
            return text.lower()
        else:
            return str(text).lower()

    def train_classifier(self, reviews, labels):
        vectorizer = CountVectorizer(preprocessor=self.preprocess_text)
        X = vectorizer.fit_transform(reviews)

        if X.shape[0] != len(labels):
            raise ValueError("Number of samples in features and labels must be the same.")

        self.clf = MultinomialNB()
        self.clf.fit(X, labels)
        return make_pipeline(vectorizer, self.clf)

    def transform_scale(self, score):
        return 5 * score + 5  # Convert the sentiment score from -1 to 1 scale to 0 to 10 scale

    def analyze_sentiment(self, reviews):
        sentiments = [{'compound': self.transform_scale(self.sia.polarity_scores(str(review))["compound"]),
                       'pos': self.sia.polarity_scores(str(review))["pos"],
                       'neu': self.sia.polarity_scores(str(review))["neu"],
                       'neg': self.sia.polarity_scores(str(review))["neg"]}
                      for review in reviews if isinstance(review, str)]
        return sentiments

    def calculate_overall_sentiment(self, reviews):
        compound_scores = [self.sia.polarity_scores(str(review))["compound"] for review in reviews if
                           isinstance(review, str)]
        overall_sentiment = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        return self.transform_scale(overall_sentiment)

    def analyze_periodic_sentiment(self, reviews, period):
        period_reviews = [' '.join(reviews[i:i + period]) for i in range(0, len(reviews), period)]
        return self.analyze_sentiment(period_reviews)

    def interpret_sentiment(self, sentiments):
        avg_sentiment = sum([sentiment['compound'] for sentiment in sentiments]) / len(sentiments) if sentiments else 0
        if avg_sentiment >= 6.5:
            description = "Excellent progress, keep up the good work!"
        elif avg_sentiment >= 6.2:
            description = "Good progress, continue to work hard!"
        else:
            description = "Needs improvement, stay motivated and keep trying!"

        trend = "No change"
        if len(sentiments) > 1:
            first_half_avg = sum([sentiment['compound'] for sentiment in sentiments[:len(sentiments) // 2]]) / (
                        len(sentiments) // 2)
            second_half_avg = sum([sentiment['compound'] for sentiment in sentiments[len(sentiments) // 2:]]) / (
                        len(sentiments) // 2)
            if second_half_avg > first_half_avg:
                trend = "Improving"
            elif second_half_avg < first_half_avg:
                trend = "Declining"

        return description, trend


st.title("Student Review Sentiment Analysis")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file")

if uploaded_file:
    df = pd.read_csv(io.BytesIO(uploaded_file.read()), encoding='utf-8')
    st.write(df.head())  # Debug statement to check the loaded data
    analyzer = SentimentAnalyzer()
    if 'teaching' in df.columns:
        teaching_reviews = df['teaching'].dropna().astype(str).tolist()
        teaching_sentiments = [analyzer.analyze_sentiment(review) for review in teaching_reviews]

        # Calculate overall sentiment score
        overall_teaching_sentiment = sum(teaching_sentiments) / len(teaching_sentiments)


        # Plotting sentiment
        weeks = list(range(1, len(sentiments) + 1))
        sentiment_scores = [sentiment['compound'] for sentiment in sentiments]
        pos_scores = [sentiment['pos'] for sentiment in sentiments]
        neu_scores = [sentiment['neu'] for sentiment in sentiments]
        neg_scores = [sentiment['neg'] for sentiment in sentiments]

        fig, ax = plt.subplots()
        ax.plot(weeks, sentiment_scores, label="Overall", color="blue")
        ax.fill_between(weeks, sentiment_scores, color="blue", alpha=0.1)
        ax.plot(weeks, pos_scores, label="Positive", color="green")
        ax.plot(weeks, neu_scores, label="Neutral", color="gray")
        ax.plot(weeks, neg_scores, label="Negative", color="red")

        ax.set_xlabel('Week')
        ax.set_ylabel('Sentiment Score')
        ax.set_title('Sentiment Analysis')
        ax.legend()
        st.pyplot(fig)

        description, trend = analyzer.interpret_sentiment(sentiments)
        st.subheader("Progress Description")
        st.write(f"Sentiment Trend: {trend}")
        st.write(f"Description: {description}")

        # Breakdown of analysis
        st.subheader("Breakdown of Analysis")
        breakdown_df = pd.DataFrame(sentiments, index=list(range(1, len(sentiments) + 1)))
        st.write(breakdown_df)

        # Train Naive Bayes classifier
        st.subheader("Naive Bayes Classifier")
        reviews = df[feedback_columns].values.flatten().tolist()
        labels = [1 if s['compound'] >= 0.65 else 0 for column in feedback_columns for s in sentiments[column]]
        pipeline = analyzer.train_classifier(reviews, labels)
        st.write("Classifier trained successfully.")

        # Prediction on new data
        test_reviews = st.text_area("Enter reviews for prediction (separate each review with a new line):")
        if test_reviews:
            test_reviews_list = test_reviews.split('\n')
            predictions = pipeline.predict(test_reviews_list)
            st.write("Predictions:")
            st.write(predictions)
   
        else:
            st.write("Columns mismatch. Please ensure the CSV file contains the required columns.")
