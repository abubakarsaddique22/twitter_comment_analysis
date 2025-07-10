import numpy as np 
import pandas as pd 
import re
import nltk
import string
import os
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

def load_data(train_data,test_data):
    train_data=pd.read_csv(train_data)
    test_data=pd.read_csv(test_data)
    return test_data,test_data


def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))
    
    # Lowercase
    text = text.lower()
    
    # Remove stop words
    text = " ".join([word for word in text.split() if word not in stop_words])
    
    # Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])
    
    # Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', '')
    text = re.sub('\s+', ' ', text).strip()
    
    # Remove URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    text = url_pattern.sub('', text)
    
    # Lemmatization
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
    
    return text

def save_data(data_path,train_data,test_data):
    os.makedirs(data_path)
    train_data.to_csv(os.path.join(data_path,'train_processed.csv'))
    test_data.to_csv(os.path.join(data_path,'test_processed.csv'))

def main():
    nltk.download('wordnet')
    nltk.download('stopwords')
    train_data, test_data = load_data('data/raw/train.csv', 'data/raw/test.csv')

    # Apply text cleaning to "content" column
    train_data['content'] = train_data['content'].apply(clean_text)
    test_data['content'] = test_data['content'].apply(clean_text)

    # Save the cleaned data
    data_path = os.path.join('data', 'processed')
    save_data(data_path, train_data, test_data)

# ----------------------------------------
if __name__ == '__main__':
    main()