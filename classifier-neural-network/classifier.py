#tar -cJf filename.tar.xz /path/to/folder_or_file ...
#https://blog.mimacom.com/text-classification/

from keras.datasets import imdb
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras import layers
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
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, 
                            output_dim=embedding_dim, 
                            input_length=maxlen))
    model.add(layers.Flatten())
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(output_size, activation='sigmoid'))
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    model.summary()
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

    multilabel_binarizer = MultiLabelBinarizer()
    y = multilabel_binarizer.fit_transform(df.tags)
    sentences = df['content'].values


    tokenizer = load_tokenizer(sentences)
    maxlen = 100

    X_train, X_test, y_train, y_test = create_train_and_test_data(tokenizer, sentences, y, maxlen)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
    embedding_dim = 50
    output_size = len(y[0])

    model = create_model(vocab_size, embedding_dim, maxlen, output_size)

    history = model.fit(X_train, y_train,
                        epochs=1,
                        verbose=True,
                        validation_data=(X_test, y_test),
                        batch_size=10)
    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


    save_model(model)
    #print(multilabel_binarizer.inverse_transform(np.array([y[0]])))

    #test using new data
    sentence_test = ["La entrada en el 2020 ha comportado cambios en la edad de jubilación y en el cálculo de los años cotizados que se tienen en cuenta para determinar la prestación. Las medidas son de carácter automático, ya que forman parte de la reforma de 2011 que hace que la edad para jubilarse se retrase paulatinamente hasta llegar a los 67 años.  ADVERTISING  Todo esto se da mientras resta pendiente saber cuál será la subida de las prestaciones en el 2020, ya que aunque el Gobierno en funciones ha prometido que se subirán el 0,9% y no perderán poder adquisitivo, la medida no se tomará hasta que esté formado un Ejecutivo. En diciembre de 2019 en España se contabilizaban 6.089.294 pensiones de jubilación, con una prestación media de 1.143,55 euros mensuales.   Pensiones en 2020 Los cambios en la edad de jubilación  Respecto a la edad de jubilación, cada año se va retrasando en virtud del régimen establecido en la reforma de 2011 aprobada durante el mandato de José Luis Rodríguez Zapatero. De esta forma, en 2020 la edad legal ordinaria será de 65 años y 10 meses. Esta edad se aplicará a aquellos que han cotizado menos de 37 años.  Si una persona llega a los 65 años en 2020 y ha cotizado 37 años o más, ya podrá jubilarse con 65 años.  En el caso de la jubilación parcial, en la que se combina trabajo y prestación, el mínimo será de 61 años y 10 meses con 35 años o más cotizados; o de 62 años y 8 meses con 33 años cotizados.  Con cada año que pasa es necesaria más edad para acceder a la jubilación, tanto si se ha cotizado por encima o por debajo de los periodos de referencia  Con cada año que pasa es necesaria más edad para acceder a la jubilación, tanto si se ha cotizado por encima o por debajo de los periodos de referencia Pensiones en 2020 Los cambios en el cálculo de la pensión  Por lo que respecta al cálculo de la pensión que se cobrará la momento de jubilarse, en 2020 se tendrán en cuenta los últimos 23 años cotizados. Estos años cotizados conforman la base reguladora, que es la suma de las bases de cotización en dicho periodo. Hay que tener en cuenta que cuantos más años se tengan en cuenta es posible que se recorte más la pensión, ya que en los últimos años de vida laboral es cuando mejores salarios se suelen cobrar.   Esta es otra de las reformas introducidas con los cambios en las pensiones de la década anterior, momento hasta el que se tenían en cuenta los últimos 15 años trabajados. La idea es que para 2022 ya se tengan en cuenta los últimos 25 años cotizados. De esta manera, en 2021 se computarán los últimos 24 años trabajados y en 2022 los últimos 25 años cotizados.  La base reguladora de la pensión se obtiene de dividir los meses de los años exigidos por el divisor correspondiente La base reguladora de la pensión se obtiene de dividir los meses de los años exigidos por el divisor correspondiente (LV) En 2023 El recorte de las pensiones que viene  Otra de las medidas que tendrán un fuerte calado en el sistema es la llegada del factor de sostenibilidad, que se aplicará a partir de 2023 e irá recortando las nuevas pensiones, teniendo en cuenta que los pensionistas vivirán más. Dicha medida en un principio debía aplicarse en 2019.  El conjunto de medidas se puede consultar al detalle en la guía para la jubilación del Ministerio de Trabajo, Migraciones y Seguridad Social."]
    xnew = tokenizer.texts_to_sequences(sentence_test)
    xnew = pad_sequences(xnew, padding='post', maxlen=maxlen)

    ynew = model.predict_classes(xnew)
    print(ynew)
    print(multilabel_binarizer.inverse_transform(np.array([ynew])))
    


if __name__== "__main__":
  main()