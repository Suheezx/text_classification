import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Single text preprocessing"""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split() if t not in stop_words]
    return " ".join(tokens)

def preprocess_series(texts):
    """List of texts preprocessing"""
    return [preprocess_text(t) for t in texts]
