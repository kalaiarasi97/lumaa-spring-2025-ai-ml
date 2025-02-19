import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure necessary NLTK packages are downloaded
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

## Data Loading

#importing Downloaded file from Kaggle
df = pd.read_csv("wiki_movie_plots_deduped.csv")
print(df.head())  # Display first few rows

# Remove duplicate rows
df_deduped = df.drop_duplicates()

# Select relevant columns
df = df[['Title', 'Genre', 'Plot']]

# Drop rows with missing values in 'Plot' or 'Genre'
df = df.dropna(subset=['Plot', 'Genre'])

# Reduce dataset size for quick processing (random 200 samples)
df_sample = df.sample(n=200, random_state=42).reset_index(drop=True)
print(df_sample.head())

# Save the cleaned dataset to a new CSV file
csv_output_path = "sampled_movie_dataset.csv"
df_sample.to_csv(csv_output_path, index=False)


## Data Pre-Processing

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))
print("Stopwords loaded:", len(stop_words))  # Check if stopwords are loaded properly

def clean_text(text):
    print("Original:", text)  # Debugging line
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    cleaned_text = " ".join(words)
    return cleaned_text


print("Checking dataset columns:", df.columns)
print("Initial dataset shape:", df.shape)
print("Dataset after deduplication:", df_deduped.shape)
print("Dataset after dropping NaNs:", df.shape)
print("Sampled dataset shape:", df_sample.shape)

print("Cleaning text...")
df_sample["Cleaned_Plot"] = df_sample["Plot"].apply(clean_text)
print("Cleaning complete! Sample output:")

#Create TF-IDF vectors and compute cosine similarity
df_movies = df_sample[["Title", "Cleaned_Plot"]]

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df_movies['Cleaned_Plot'])

# Function to get movie recommendations
def recommend_movies(description, top_n=5):
    query_vec = tfidf_vectorizer.transform([description])  # Transform input into TF-IDF
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]  # Get top N indices
    
    recommendations = []
    for idx in top_indices:
        recommendations.append((df_movies.iloc[idx]['Title'], similarity_scores[idx]))
    
    return recommendations

# Example usage 1
user_input1 = "I like action movies set in space with aliens"
recommended_movies1 = recommend_movies(user_input1, top_n=3)

# Example usage 2
user_input2 = "I enjoy movies about time travel and complex narratives"
recommended_movies2 = recommend_movies(user_input2, top_n=3)

# Combine results
print("Recommended Movies:")
print(f"For input: '{user_input1}'")
for title, score in recommended_movies1:
    print(f"{title} (Similarity: {score:.4f})")

print(f"\nFor input: '{user_input2}'")
for title, score in recommended_movies2:
    print(f"{title} (Similarity: {score:.4f})")
