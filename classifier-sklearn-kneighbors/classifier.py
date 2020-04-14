#tar -cJf filename.tar.xz /path/to/folder_or_file ...
#https://blog.mimacom.com/text-classification/

from keras.datasets import imdb
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier

from joblib import dump, load

nltk.download('stopwords')
import pandas as pd
import glob
import re
import numpy as np
import pickle



class Classifier:
    def __init__(self):
        self.REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile('[^\w\s]')
        self.STOPWORDS = set(stopwords.words('spanish'))
        self.tokenizer = None
        self.multilabel_binarizer = MultiLabelBinarizer()
        self.model = None
        self.maxlen = 100


    def clean_text(self, text):
        text = text.lower() # lowercase text
        text = self.REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
        text = self.BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    #    text = re.sub(r'\W+', '', text)
        text = ' '.join(word for word in text.split() if word not in self.STOPWORDS) # remove stopwors from text
        return text

    def clean_text_in_tags(self, tags):
        clean_tags = []
        for tag in tags:
            clean_tags = clean_tags + [self.clean_text(tag)]
        return clean_tags
        

    def clean_news(self, df):
        print("cleaning the text data")
        df = df.reset_index(drop=True)
        df.dropna(subset=['tags'], inplace=True)
        df['tags'] = df['tags'].apply(self.clean_text_in_tags)
        df['content'] = df['content'].apply(self.clean_text)
        df['content'] = df['content'].str.replace('\d+', '')
        return df

    def create_tags_and_multilabel_biniarizer(self, df):
        print("creating tags and tag index for classes")
        y = self.multilabel_binarizer.fit_transform(df.tags)
        # Serialize both the pipeline and binarizer to disk.
        with open('../data/neural_network_config/multilabel_binarizer.pickle', 'wb') as f:
            pickle.dump((self.multilabel_binarizer), f, protocol=pickle.HIGHEST_PROTOCOL)
        return y
        
    def load_tokenizer(self, sentences):
        print("loading toikenizer")
        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(sentences)

        # saving tokenizer
        with open('../data/neural_network_config/tokenizer.pickle', 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_train_and_test_data(self, sentences, y):
        print("separating data into test data and train data")
        sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

        X_train = self.tokenizer.texts_to_sequences(sentences_train)
        X_test = self.tokenizer.texts_to_sequences(sentences_test)

        X_train = pad_sequences(X_train, padding='post', maxlen=self.maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=self.maxlen)
        return X_train, X_test, y_train, y_test

    def create_model(self):
        print("creating model")

    
        #self.model = Sequential()
        #self.model.add(Embedding(vocab_size, 20, input_length=self.maxlen))
        #self.model.add(Dropout(0.1))
        #self.model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
        #self.model.add(GlobalMaxPool1D())
        #self.model.add(Dense(output_size))
        #self.model.add(Activation('sigmoid'))
        #self.model.compile(optimizer=Adam(0.015), loss='binary_crossentropy', metrics=['categorical_accuracy'])
        
        self.model = MultiOutputClassifier(KNeighborsClassifier())

    def save_model(self):
        print("saving model")
        # saving model
        dump(self.model, '../data/neural_network_config/model-sklearn-kneighbors.joblib') 
    
    def create_and_train_model(self):
        filename = "../data/json_news_tagged_bundle/clean_data-unified-tags.json"
        df = pd.read_json(filename)
        df = self.clean_news(df)

        y = self.create_tags_and_multilabel_biniarizer(df)
        sentences = df['content'].values


        self.load_tokenizer(sentences)

        X_train, X_test, y_train, y_test = self.create_train_and_test_data(sentences, y)

        self.create_model()
        
        history = self.model.fit(X_train, y_train)

        print(history)

        self.save_model()


if __name__== "__main__":
    classifier = Classifier()
    classifier.create_and_train_model()