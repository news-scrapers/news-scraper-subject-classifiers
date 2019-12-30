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

nltk.download('stopwords')
import pandas as pd
import glob
import re
import numpy as np
import pickle

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
    print("cleaning the text data")
    df = df.reset_index(drop=True)
    df.dropna(subset=['tags'], inplace=True)
    df['tags'] = df['tags'].apply(clean_text_in_tags)
    df['content'] = df['content'].apply(clean_text)
    df['content'] = df['content'].str.replace('\d+', '')
    return df

def create_tags_and_multilabel_biniarizer(df):
    multilabel_binarizer = MultiLabelBinarizer()
    y = multilabel_binarizer.fit_transform(df.tags)
    # Serialize both the pipeline and binarizer to disk.
    with open('../data/neural_network_config/multilabel_binarizer.pickle', 'wb') as f:
        pickle.dump((multilabel_binarizer), f, protocol=pickle.HIGHEST_PROTOCOL)
    return y, multilabel_binarizer
    
def load_tokenizer(sentences):
    print("loading toikenizer")
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sentences)

    # saving tokenizer
    with open('../data/neural_network_config/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return tokenizer

def create_train_and_test_data(tokenizer, sentences, y, maxlen):
    print("separating data into test data and train data")
    sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000)

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    return X_train, X_test, y_train, y_test

def create_model(vocab_size, embedding_dim, maxlen, output_size):
    print("creating model")
    filter_length = 300

    #model = Sequential()
    #model.add(Embedding(vocab_size, 20, input_length=maxlen))
    #model.add(Dropout(0.15))
    #model.add(GlobalMaxPool1D())
    #model.add(Dense(output_size, activation='sigmoid'))

    model = Sequential()
    model.add(Embedding(vocab_size, 20, input_length=maxlen))
    model.add(Dropout(0.1))
    model.add(Conv1D(filter_length, 3, padding='valid', activation='relu', strides=1))
    model.add(GlobalMaxPool1D())
    model.add(Dense(output_size))
    model.add(Activation('sigmoid'))


    model.compile(optimizer=Adam(0.015), loss='binary_crossentropy', metrics=['categorical_accuracy'])
    
    return model

def save_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("../data/neural_network_config/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("../data/neural_network_config/model.h5")
    print("Saved model to disk")

def main():
    filename = "../data/json_news_tagged_bundle/bundle.json"
    df = pd.read_json(filename)
    df = clean_news(df)

    y, multilabel_binarizer = create_tags_and_multilabel_biniarizer(df)
    sentences = df['content'].values


    tokenizer = load_tokenizer(sentences)
    maxlen = 100

    X_train, X_test, y_train, y_test = create_train_and_test_data(tokenizer, sentences, y, maxlen)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    embedding_dim = 50
    output_size = len(y[0])

    model = create_model(vocab_size, embedding_dim, maxlen, output_size)
    
    callbacks = [
    ReduceLROnPlateau(),
    EarlyStopping(patience=4),
    ModelCheckpoint(filepath='model-simple.h5', save_best_only=True)]

    history = model.fit(X_train, y_train,
                        epochs=20,
                        batch_size=32,
                        validation_data=(X_test, y_test),
                        callbacks=callbacks)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


    save_model(model)


if __name__== "__main__":
  main()