import os
import sys

import pickle
import numpy as np
import warnings

from tensorflow import keras
from keras import preprocessing, layers, models
from nltk.corpus import stopwords

class MLKeras:
  def __init__(
        self, 
        filename      : str   = 'data/keras-data.pickle',
        optimize      : bool  = True,
        maxlen        : int   = None 
      ) -> None:
    self.__filename = os.path.join(sys.path[0], filename)
    self.__maxlen = maxlen
    self.__optimize = optimize

    self.__initialize_data()

  def __initialize_data(self) -> None:
    """
    Unpickles data and initializes the training and test-data
    into arrays that could be trained on
    """
    data = self.__unpickle_data()

    raw_x_train = data["x_train"]
    raw_y_train = data["y_train"]

    raw_x_test = data["x_test"]
    raw_y_test = data["y_test"]

    self.__vocabular_size = data["vocab_size"]

    # Padding the sequences to match a given length
    if self.__maxlen is None and not self.__optimize:
      warnings.warn("No length is assigned for padding! Will result in slower computation.")
      self.__maxlen = data["max_length"]
    elif self.__maxlen is None:
      # Using the mean to determine the optimal length
      self.__maxlen = np.array([len(x_t) for x_t in raw_x_train]).mean()

    # It is assumed that the initial values are most important. People will
    # generally tend to express the most important first, especially when 
    # giving such reviews. Therefore, using post-padding, as it will likely
    # maintain more of the information in the sentences. Would be interesting
    # to compare the difference between pre and post-padding 

    # Exception: np.float64 cannot be interpreted as an integer
    self.__x_train = preprocessing.sequence.pad_sequences(
      sequences=raw_x_train, 
      maxlen=self.__maxlen,
      dtype='int32',
      padding='post'
    )
    self.__x_test = preprocessing.sequence.pad_sequences(
      sequences=raw_x_test,
      maxlen=self.__maxlen,
      dtype='int32',
      padding='post'
    )

    self.__y_train = raw_y_train
    self.__y_test = raw_y_test

  def __unpickle_data(self) -> dict:
    """
    Extracting data from a pickle-file
    
    Returns dictionary with 4 keys: {x_train, y_train, x_text, y_test}
    """

    data = None
    with open(file=self.__filename, mode="rb") as file:
      data = pickle.load(file) 

    assert data is not None, "No data read"
    return data   
  
  def __create_model(self) -> keras.models.Sequential:
    """
    Creates and returns a keras-model as RNN (LSTM)-network

    According to my understanding of the task, the model should
    consist of 
      -an embedding layer
      -a LSTM layer
      -a dense layer
    and one should thereafter only play around with the parameters
    for the different layers.

    The thing that bother me, is the dimensionality of the problem.
    I cannot quite understand how to set up the input and output 
    dimensionality correctly. Since it is only three layers in the model, 
    this will simplify things a bit. However, I cannot quite understand 
    how the information flow will be best in the model.
    Also, when questioning the dimensionaility of the information flow,
    how is the input structured? I cannot quite understand the meaning
    of the input data and how to best use it. Pherhaps that is the intention
    of the assignment however...

    Have written some documentation for each of the layers, as well as 
    the corresponding links to the documentation. This is to simplify
    looking up the information later
    """
    model = models.Sequential()

    # Information regarding the size of the model - uncertain how 
    # to quantify these values...
    output_embedded_dim = 64
    lstm_units = 32
    dense_units = 16

    # Embedded layer will turn the indeces into a dense vector of 
    # fixed size. Requires a fixed range on the input
    # https://keras.io/api/layers/core_layers/embedding/
    model.add(
      layers.Embedding(
        input_dim=self.__vocabular_size, 
        output_dim=output_embedded_dim
      )
    )

    # Using the default-values to make it more likely to use the GPU
    # https://keras.io/api/layers/recurrent_layers/lstm/
    model.add(
      layers.LSTM(
        units=lstm_units
      )
    )

    # Dense layer will implement an activation + bias function.
    # https://keras.io/api/layers/core_layers/dense/
    model.add(
      layers.Dense(
        units=dense_units,
        activation="softmax" # To get output to sum to 1
      )
    )

    # Using the parameters given in the example on blackboard
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    model.summary()

    return model

  def lstm(
        self,
        num_epochs: int = 10
      ) -> tuple:
    """
    Trains and tests a lstm. Returns normalized 
    values indicating how well the training and the testing performed

    Returns a tuple containing:
      -training_accuracy : Results from the training-set. Normalized to range [0, 1]
      -testing_accuracy  : Results from the testing-set. Normalized to range [0, 1]
    """
    model = self.__create_model()

    model.fit(
      x=self.__x_train,
      y=self.__y_train,
      batch_size=None, # Do not specify the batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances (since they generate batches). 
      epochs=num_epochs,
      verbouse=1
    )

    return model.evaluate(
      x=self.__x_test,
      y=self.__y_test,
      batch_size=None,
      verbouse=1
    )

if __name__ == '__main__':
  ml_keras = MLKeras()
