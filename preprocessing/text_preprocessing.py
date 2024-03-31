
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
import ssl
from nltk.stem import WordNetLemmatizer

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    """Clean text by removing HTML tags, URLs, hashtags, mentions, non-alphanumeric characters, punctuation and converting to lowercase."""
    text = re.sub(r'<.*?>', '', text) 
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) 
    text = re.sub(r'@\w+', '', text)  
    text = re.sub(r'#\w+', '', text)  
    text = re.sub(r'\W', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))  
  
    text = text.lower()  
    return text

def remove_stopwords(text):
    """Remove stopwords from the text."""
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return ' '.join(filtered_text)


def lemmatize_words(text):
    """Lemmatize the words in the text."""
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = ' '.join([lemmatizer.lemmatize(word) for word in word_tokens])
    return lemmatized_text

def preprocess_data(df, text_column_name='text',target_column_name = 'sentiment', columns_to_remove=[]):
    """
    Apply all preprocessing steps to the dataframe.
    Args:
        df (pandas.DataFrame): DataFrame containing the text to preprocess.
        text_column_name (str): Name of the column containing the text data.
        columns_to_remove (list) : Columns to be removed.
    Returns:
        pandas.DataFrame: DataFrame with preprocessed text in the specified column.
    """
    df[text_column_name] = df[text_column_name].astype(str)
    df[text_column_name] = df[text_column_name].apply(clean_text)
    df[text_column_name] = df[text_column_name].apply(remove_stopwords)
    df[text_column_name] = df[text_column_name].apply(lemmatize_words)
    df[target_column_name] = df[target_column_name].map({"negative": 0, "neutral": 1, "positive": 2})

    for column in columns_to_remove:
        if column in df.columns:
            df = df.drop(columns=[column])
    df = df.dropna()

    return df
