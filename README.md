#Code for Part 1
# Step 1: Import Libraries
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

# Step 2: Load the dataset
df = pd.read_csv("/content/sample_data/IMDB Dataset.csv")

# Step 3: Explore the dataset
print("Dataset Info:\n", df.info())
print("\nMissing Values:\n", df.isnull().sum())
print("\nLabel Distribution:\n", df['sentiment'].value_counts())

# Step 4: Text Preprocessing Function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and numbers
    tokens = text.split()  # Tokenize (split by space)
    tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]  # Remove stopwords
    return ' '.join(tokens)

# Step 5: Apply preprocessing (use a sample if full dataset is too large)
df_sample = df.head(1000).copy()
df_sample['cleaned_review'] = df_sample['review'].apply(preprocess_text)

# Step 6: Vectorization - CountVectorizer
count_vectorizer = CountVectorizer(max_features=1000)
X_count = count_vectorizer.fit_transform(df_sample['cleaned_review'])

# Step 7: Vectorization - TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(df_sample['cleaned_review'])

# Step 8: Output Vector Shapes and Sample Features
print("\nCountVectorizer Shape:", X_count.shape)
print("CountVectorizer Features (first 10):", count_vectorizer.get_feature_names_out()[:10])

print("\nTF-IDF Shape:", X_tfidf.shape)
print("TF-IDF Features (first 10):", tfidf_vectorizer.get_feature_names_out()[:10])
