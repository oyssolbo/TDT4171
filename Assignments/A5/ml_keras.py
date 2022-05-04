import os
import sys

import pickle
import numpy as np
import warnings
import math

from tensorflow import keras
from keras import preprocessing, layers, models, utils

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
    print("Unpickling data")
    data = self.__unpickle_data()

    # Converting the data into integers to be trained on
    raw_x_train = data["x_train"]
    raw_y_train = data["y_train"]

    raw_x_test = data["x_test"]
    raw_y_test = data["y_test"]

    self.__vocabular_size = data["vocab_size"]

    # Creating the lists into list of lists of integers
    raw_x_train = [[int(x) for x in x_list] for x_list in raw_x_train]
    raw_x_test = [[int(x) for x in x_list] for x_list in raw_x_test]

    # raw_y_train = [int(y) for y in raw_y_train]
    # raw_y_test = [int(y) for y in raw_y_test]

    # Padding the sequences to match a given length
    if self.__maxlen is None and not self.__optimize:
      warnings.warn("No length is assigned for padding! Will result in slower computation.")
      self.__maxlen = math.ceil(data["max_length"])
    elif self.__maxlen is None:
      # Using the mean to determine the optimal length
      self.__maxlen = math.ceil(np.array([len(x_t) for x_t in raw_x_train]).mean())

    # It is assumed that the initial values are most important. People will
    # generally tend to express the most important first, especially when 
    # giving such reviews. Therefore, using post-padding, as it will likely
    # maintain more of the information in the sentences. Would be interesting
    # to compare the difference between pre and post-padding 
    print("Padding data")
    self.__x_train = preprocessing.sequence.pad_sequences(
      sequences=raw_x_train, 
      maxlen=self.__maxlen,
      padding='post'
    )
    self.__x_test = preprocessing.sequence.pad_sequences(
      sequences=raw_x_test,
      maxlen=self.__maxlen,
      padding='post'
    )

    self.__y_train = np.array(raw_y_train, dtype=int)
    self.__y_test = np.array(raw_y_test, dtype=int)

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
    """
    print("Building sequential model")
    model = models.Sequential()

    # Information regarding the size of the model - uncertain how 
    # to quantify these values...
    output_embedded_dim = 64
    lstm_units = 32
    first_dense_units = 16
    second_dense_units = 16
    third_dense_units = 8

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

    # Extra layers added for science
    # model.add(
    #   layers.Dense(
    #     units=first_dense_units,
    #     activation="tanh" 
    #   )
    # )

    # model.add(
    #   layers.Dense(
    #     units=second_dense_units,
    #     activation="tanh" 
    #   )
    # )

    # model.add(
    #   layers.Dense(
    #     units=third_dense_units,
    #     activation="tanh" 
    #   )
    # )

    # model.add(
    #   layers.Dense(
    #     units=4 
    #   )
    # )

    # model.add(
    #   layers.Dense(
    #     units=2
    #   )
    # )

    model.add(
      layers.Dense(
        units=1,
        activation="tanh" 
      )
    )

    # https://www.tensorflow.org/api_docs/python/tf/keras/losses
    model.compile(
      optimizer='adam',
      loss='huber',  
      metrics=['accuracy']
    )
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

    print("Fitting data for {} epochs".format(num_epochs))
    model.fit(
      x=self.__x_train,
      y=self.__y_train,
      batch_size=None,  
      epochs=num_epochs,
      verbose=1
    )

    print("Evaluating model for {} epochs".format(num_epochs))
    return model.evaluate(
      x=self.__x_test,
      y=self.__y_test,
      batch_size=None,
      verbose=1
    )

if __name__ == '__main__':
  ml_keras = MLKeras()
  print(ml_keras.lstm(num_epochs=10))
