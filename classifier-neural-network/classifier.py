#tar -cJf filename.tar.xz /path/to/folder_or_file ...
#https://blog.mimacom.com/text-classification/

from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, Embedding, Flatten,MaxPooling1D, GlobalMaxPool1D, Dropout, Conv1D
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

import pandas as pd
import glob
import re
import numpy as np
import pickle



class Classifier:
    def __init__(self):
        print("loading toikenizer")
        with open('../data/neural_network_config/tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        self.multilabel_binarizer = MultiLabelBinarizer()
        self.model = None
        self.maxlen = 700


    def create_tags_and_multilabel_biniarizer(self, df):
        print("creating tags and tag index for classes")
        y = self.multilabel_binarizer.fit_transform(df.common_tags)
        # Serialize both the pipeline and binarizer to disk.
        with open('../data/neural_network_config/multilabel_binarizer.pickle', 'wb') as f:
            pickle.dump((self.multilabel_binarizer), f, protocol=pickle.HIGHEST_PROTOCOL)
        return y
                

    def create_train_and_test_data(self, sentences, y):
        print("separating data into test data and train data")
        sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, y, test_size=0.25, random_state=1000)

        X_train = self.tokenizer.texts_to_sequences(sentences_train)
        X_test = self.tokenizer.texts_to_sequences(sentences_test)

        X_train = pad_sequences(X_train, padding='post', maxlen=self.maxlen)
        X_test = pad_sequences(X_test, padding='post', maxlen=self.maxlen)
        return X_train, X_test, y_train, y_test

    def create_model(self, vocab_size, output_size):
        print("creating model")
        filter_length = 300

        #model = Sequential()
        #model.add(Embedding(vocab_size, 20, input_length=maxlen))
        #model.add(Dropout(0.15))
        #model.add(GlobalMaxPool1D())
        #model.add(Dense(output_size, activation='sigmoid'))
        
        self.model = Sequential()
        self.model.add(Embedding(vocab_size, 20, input_length=self.maxlen))
        self.model.add(Dropout(0.1))
        self.model.add(Conv1D(filter_length, 5, activation='relu'))
        self.model.add(Conv1D(filter_length, 5, activation='relu'))
        self.model.add(MaxPooling1D(5))
        self.model.add(Conv1D(filter_length, 5, activation='relu'))
        self.model.add(Conv1D(filter_length, 5, activation='relu'))
        self.model.add(GlobalMaxPool1D())
        self.model.add(Dense(output_size, activation="relu"))
        self.model.add(Activation('softmax'))   

        #self.model = Sequential()
        #self.model.add(Embedding(vocab_size, 20, input_length=self.maxlen))
        #self.model.add(Dense(60, activation="relu"))
        #self.model.add(Dense(60, activation="relu"))
        #self.model.add(Flatten())
        #self.model.add(Dense(output_size, activation="relu"))
        #self.model.add(Activation('softmax'))
        # create model

        self.model.compile(optimizer="rmsprop", loss='binary_crossentropy', metrics=['accuracy'])


    def save_model_structure(self):
        print("saving model")
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("../data/neural_network_config/model.json", "w") as json_file:
            json_file.write(model_json)

    def save_weights(self):
        # serialize weights to HDF5
        self.model.save_weights("../data/neural_network_config/model_new.h5")
        print("Saved model to disk")

    def create_and_train_model(self):
        filename = "../data/json_news_tagged_bundle/clean_data-unified-tags.json"
        df = pd.read_json(filename)

        y = self.create_tags_and_multilabel_biniarizer(df)
        sentences = df['content'].values

        X_train, X_test, y_train, y_test = self.create_train_and_test_data(sentences, y)

        vocab_size = len(self.tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
        output_size = len(y[0])

        self.create_model(vocab_size, output_size)
        self.save_model_structure()

        callbacks = [
        ReduceLROnPlateau(),
        EarlyStopping(patience=4),
        ModelCheckpoint(filepath='../data/neural_network_config/temp-model-new.h5', save_best_only=True)]

        history = self.model.fit(X_train, y_train,
                            epochs=40,
                            batch_size=40,
                            validation_data=(X_test, y_test),
                            callbacks=callbacks)

        loss, accuracy = self.model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))

        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))


        self.save_weights()

if __name__== "__main__":
    classifier = Classifier()
    classifier.create_and_train_model()