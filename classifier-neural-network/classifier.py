#tar -cJf filename.tar.xz /path/to/folder_or_file ...
#https://blog.mimacom.com/text-classification/

from keras.datasets import imdb
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer

nltk.download('stopwords')
import pandas as pd
import glob
import re

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^\w\s]')
STOPWORDS = set(stopwords.words('spanish'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
#    text = re.sub(r'\W+', '', text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text

def clean_text_in_tags(tags):
    clean_tags = []
    for tag in tags:
        clean_tags = clean_tags + [clean_text(tag)]
    return clean_tags
    

def clean_news(df):
    df = df.reset_index(drop=True)
    df.dropna(subset=['tags'], inplace=True)
    df['tags'] = df['tags'].apply(clean_text_in_tags)
    df['content'] = df['content'].apply(clean_text)
    df['content'] = df['content'].str.replace('\d+', '')
    return df

def main():
    filename = "../data/json_news_tagged_bundle/bundle.json"
    df = pd.read_json(filename)
    df = clean_news(df)

    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(df.tags)
    y = multilabel_binarizer.classes_    
    sentences = df['content'].values
    print(y)


if __name__== "__main__":
  main()