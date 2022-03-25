import os
import sys

import pickle
import numpy as np
import warnings
import math

from tensorflow import keras
from keras import preprocessing, layers, models, utils

"""
Debugging:
https://stackoverflow.com/questions/61550026/valueerror-shapes-none-1-and-none-3-are-incompatible
"""

"""
Log / discussion for the testing + some results I would like to keep:

First iteration running the initial values
loss: 0.0000e+00 - accuracy: 0.706312283/12283 [==============================] - 153s 12ms/step - loss: 0.0000e+00 - accuracy: 0.7063
Evaluating model for 1 epochs
   1/4079 [..............................] - ETA: 26:40 - loss: 0.0000e+00 - accuracy: 0.781  13/4079 [..............................] - ETA: 17s - loss: 0.0000e+00 - accuracy: 0.7692 4079/4079 [==============================] - 16s 4ms/step - loss: 0.0000e+00 - accuracy: 0.7061
[0.0, 0.7061243653297424]


Second iteration, adding another dense layer
Fitting data for 1 epochs
12283/12283 [==============================] - 150s 12ms/step - loss: 0.0000e+00 - accuracy: 0.7063
Evaluating model for 1 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 0.0000e+00 - accuracy: 0.7061
[0.0, 0.7061243653297424]

Observation: adding another dense layer had no impact on the performance of the algorithm. This 
was run using the same type of parameters and activation function. A change in these could have 
affected how the system would have behaved. It might also be such that the parameters selected 
to do this training is unable to actually fulfill the desired data


Third iteration, reducing the number of dense parameters on the second dense layer
and increasing number of epochs up to 3.

Fitting data for 3 epochs
Epoch 1/3
12283/12283 [==============================] - 153s 12ms/step - loss: 0.0000e+00 - accuracy: 0.7063
Epoch 2/3
12283/12283 [==============================] - 149s 12ms/step - loss: 0.0000e+00 - accuracy: 0.7063
Epoch 3/3
12283/12283 [==============================] - 151s 12ms/step - loss: 0.0000e+00 - accuracy: 0.7063
Evaluating model for 3 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 0.0000e+00 - accuracy: 0.7061
[0.0, 0.7061243653297424]

Made no change to the accuracy... Must be the parameters then that must be changed to improve the 
performance of the algorithm


Fourth iteration, reduce to only one dense layer and check the sigmoid-activation function

Fitting data for 1 epochs
12283/12283 [==============================] - 148s 12ms/step - loss: 0.0000e+00 - accuracy: 0.3011
Evaluating model for 1 epochs
4079/4079 [==============================] - 16s 4ms/step - loss: 0.0000e+00 - accuracy: 0.2939
[0.0, 0.29387563467025757]

What the fuck happened here? Is this caused by the loss of a dense layer, or is it due to the
change in the activation function? My initial guess would be the loss of the extra layer, such that
the tree went from being long to just being wide

Edit when running the fifth iteration: looks like the test activation function was the key result
here. Might be due to softmax summing to one, which the sigmoid does not. Thus maintaining the 
most important functionality of the tree / training algorithm

Fifth iteration, add an extra dense layer 

Fitting data for 1 epochs
12283/12283 [==============================] - 155s 12ms/step - loss: 0.0000e+00 - accuracy: 0.2939
Evaluating model for 1 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 0.0000e+00 - accuracy: 0.2939
[0.0, 0.29387563467025757]


Sixth iteration, change the number of variables for the different layers. Running softmax on the 
last dense layer again

output_embedded_dim = 128
lstm_units = 64
dense_units = 32

Fitting data for 1 epochs
12283/12283 [==============================] - 198s 16ms/step - loss: 0.0000e+00 - accuracy: 0.7063
Evaluating model for 1 epochs
4079/4079 [==============================] - 24s 6ms/step - loss: 0.0000e+00 - accuracy: 0.7061

Hmmmm... FML!



Seventh iteration, adding two layers with dense layers and with default activation function

output_embedded_dim = 128
lstm_units = 64
first_dense_units = 32
second_dense_units = 16
third_dense_units = 8

Pherhaps it would be better to just use some pooling of the data?


Fitting data for 1 epochs
12283/12283 [==============================] - 198s 16ms/step - loss: 0.0000e+00 - accuracy: 0.7063
Evaluating model for 1 epochs
4079/4079 [==============================] - 24s 6ms/step - loss: 0.0000e+00 - accuracy: 0.7061
[0.0, 0.7061243653297424]


Eight iteration, trying to make the layered model more narrow 

output_embedded_dim = 64
lstm_units = 32
first_dense_units = 32
second_dense_units = 32
third_dense_units = 8

Fitting data for 1 epochs
12283/12283 [==============================] - 150s 12ms/step - loss: 0.0000e+00 - accuracy: 0.7063
Evaluating model for 1 epochs
4079/4079 [==============================] - 16s 4ms/step - loss: 0.0000e+00 - accuracy: 0.7061
[0.0, 0.7061243653297424]

I fucking hate my life


Ninth iteration, trying to set the initial embedded layer to really wide

output_embedded_dim = 512
lstm_units = 128
first_dense_units = 64
second_dense_units = 32
third_dense_units = 8

Fitting data for 1 epochs
12283/12283 [==============================] - 500s 41ms/step - loss: 0.0000e+00 - accuracy: 0.7063
Evaluating model for 1 epochs
4079/4079 [==============================] - 63s 15ms/step - loss: 0.0000e+00 - accuracy: 0.7061
[0.0, 0.7061243653297424]


Just kill me! 


Tenth iteration, changed the optimizer to 'adam' and modified dimensionality for testing

output_embedded_dim = 128
lstm_units = 128
first_dense_units = 32
second_dense_units = 32
third_dense_units = 8

Fitting data for 1 epochs
12283/12283 [==============================] - 348s 28ms/step - loss: 0.0000e+00 - accuracy: 0.7063
Evaluating model for 1 epochs
4079/4079 [==============================] - 41s 10ms/step - loss: 0.0000e+00 - accuracy: 0.7061
[0.0, 0.7061243653297424]


Theory - while waiting on the tenth iteration to finish: might try to 
optimize the batch-size. Currently, the batch-size is set as None, leaving the
fit algorithm to choose the batch-value as it sees fit. Pherhaps by setting
the batch-size, could the algorithm achieve some better results??



Eleventh iteration, increasing the batch-size to 64 instead of the default 32

Fitting data for 1 epochs
6142/6142 [==============================] - 224s 36ms/step - loss: 0.0000e+00 - accuracy: 0.7063
Evaluating model for 1 epochs
4079/4079 [==============================] - 40s 10ms/step - loss: 0.0000e+00 - accuracy: 0.7061
[0.0, 0.7061243653297424]


Twelth iteration, changing to binary_crossentropy instead of categorical_crossentropy for the loss

Fuck me I am stupiiiiiiiiiiiiid! I should be shot and do not deserve to live


Fitting data for 1 epochs
12283/12283 [==============================] - 148s 12ms/step - loss: 0.2253 - accuracy: 0.9081
Evaluating model for 1 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 0.2002 - accuracy: 0.9210
[0.20021870732307434, 0.9210361242294312]

Using the binary loss-function enables the function to classify into data or not. This makes sence
as we are in this case only looking to check if the data is available or not. 

One can see that the function has generated the loss as well, which is caused by actually having a 
proper data to observe the loss from

Interesting observation regarding the accuracy of the testing, which is better than achieved by the 
training set. In most cases, it would be the other way around


Thirteenth iteration, changin the adam loss function with the adadelta optimizer

https://keras.io/api/optimizers/


Fitting data for 1 epochs
12283/12283 [==============================] - 153s 12ms/step - loss: 0.6351 - accuracy: 0.7002
Evaluating model for 1 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 0.5903 - accuracy: 0.7061
[0.5903036594390869, 0.7061243653297424]

One can see that the adam optimizer is much better


Fourtheenth iteration, testing the adamax optimizer

Fitting data for 1 epochs
12283/12283 [==============================] - 144s 12ms/step - loss: 0.2674 - accuracy: 0.8872
Evaluating model for 1 epochs
4079/4079 [==============================] - 16s 4ms/step - loss: 0.2417 - accuracy: 0.8984
[0.24167457222938538, 0.8983666300773621]

Adam seems to achieve some better results


Fifthenth iteration, running adam and trying to reduce the comnplexity of the model. Changed back
to softmax

Fitting data for 1 epochs
12283/12283 [==============================] - 147s 12ms/step - loss: 0.2281 - accuracy: 0.7063
Evaluating model for 1 epochs
4079/4079 [==============================] - 16s 4ms/step - loss: 0.1856 - accuracy: 0.7061
[0.18564191460609436, 0.7061243653297424]

Using softmax instead of sigmoid has a huge impact on the performance of the system. Why?????
I had originally though that it would be better to use the softmax, such that all of the values
would sum to one. However that would not necessarily be the case, as the system is only 
operating on one output, and not a set of possible outputs. In that case, using the sigmoid 
function appears to work slightly better


Sixteenth iteration, testing tanh as an activation function

Fitting data for 1 epochs
12283/12283 [==============================] - 147s 12ms/step - loss: 0.2470 - accuracy: 0.8996
Evaluating model for 1 epochs
4079/4079 [==============================] - 16s 4ms/step - loss: 0.1899 - accuracy: 0.9276
[0.18989117443561554, 0.9276477098464966]


17th iteration, testing the linear activation function

Fitting data for 1 epochs
12283/12283 [==============================] - 147s 12ms/step - loss: 3.0896 - accuracy: 0.7264
Evaluating model for 1 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 4.4815 - accuracy: 0.7061
[4.481513023376465, 0.7061243653297424]


18th iteration, testing exponential activation function

Fitting data for 1 epochs
12283/12283 [==============================] - 148s 12ms/step - loss: 0.4351 - accuracy: 0.8107
Evaluating model for 1 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 0.2720 - accuracy: 0.8914
[0.2719668447971344, 0.8914409279823303]


19th iteration, tetsing swish

Fitting data for 1 epochs
12283/12283 [==============================] - 148s 12ms/step - loss: 10.8938 - accuracy: 0.2937
Evaluating model for 1 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 10.8920 - accuracy: 0.2939
[10.891979217529297, 0.29387563467025757]






Short summary regarding the testing:
It looks like the optimal results are obtained when one is running sigmoid function.

The report should focus on the different testings that are performed:
  - which output function is best with the corresponding results
  - the effect of the different optimizers and lossfunctions
  - the effect of the different layers, and how it was changed when increasing number of layers. 
      Although this is a bit difficult to answer, as I have only tested using the LSTM, dense and 
      embedded layer. There might be other layers that could increase the effectiveness of the training
  - the effect of choosing the values that are selected


20th iteration, resulting back to the sigmoid function

Fitting data for 1 epochs
12283/12283 [==============================] - 150s 12ms/step - loss: 0.2405 - accuracy: 0.9008
Evaluating model for 1 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 0.1885 - accuracy: 0.9242
[0.1884700357913971, 0.9241542220115662]



21st iteration, reducing the number of layers down to just running the embedded layer, lstm layer
and the dense layer. It is theorized that the previous iteration/test showed that this data is 
relatively equivalent. What remains is therefore to explore the choice of the activation function 
and the number of units for each layer

output_embedded_dim = 32
lstm_units = 32

Fitting data for 1 epochs
12283/12283 [==============================] - 148s 12ms/step - loss: 0.2309 - accuracy: 0.9058
Evaluating model for 1 epochs
4079/4079 [==============================] - 16s 4ms/step - loss: 0.2072 - accuracy: 0.9233
[0.20718835294246674, 0.9232961535453796]

One can see that the accuracy during training was slightly improved, however the accuracy and loss 
during testing, was relatively worse. This means that there occurs some effects that the deeper modell
is better able to capture compared to this model, and that this model will be more likely to starting
to become overfitted.



22nd iteration, doubling the number of internal layers for both the lstm and the output of the 
embedded

output_embedded_dim = 64
lstm_units = 64

Fitting data for 1 epochs
12283/12283 [==============================] - 194s 16ms/step - loss: 0.2917 - accuracy: 0.8800
Evaluating model for 1 epochs
4079/4079 [==============================] - 23s 6ms/step - loss: 0.1870 - accuracy: 0.9236
[0.186972975730896, 0.9236255884170532]

Interesting to note the loss in the training accuracy, while maintaining the testing accuracy quite
high. 

What would be the effect (just for testing) if the model contained more layers but kept the 
number of internal nodes as such?


23rd iteration, adding more layers 


output_embedded_dim = 64
lstm_units = 64
first_dense_units = 32
second_dense_units = 32
third_dense_units = 16


Fitting data for 1 epochs
12283/12283 [==============================] - 204s 16ms/step - loss: 0.2208 - accuracy: 0.9111
Evaluating model for 1 epochs
4079/4079 [==============================] - 24s 6ms/step - loss: 0.1844 - accuracy: 0.9189
[0.1843557208776474, 0.9188833236694336]



24th iteration, reducing all of the layers to have a narrow network

output_embedded_dim = 32
lstm_units = 32
first_dense_units = 16
second_dense_units = 16
third_dense_units = 8

Fitting data for 1 epochs
12283/12283 [==============================] - 149s 12ms/step - loss: 0.2217 - accuracy: 0.9097
Evaluating model for 1 epochs
4079/4079 [==============================] - 18s 4ms/step - loss: 0.1895 - accuracy: 0.9264
[0.189487487077713, 0.9264066219329834]



25th iteration, setting all activation functions to be relu instead of just pass-through

Fitting data for 1 epochs
12283/12283 [==============================] - 149s 12ms/step - loss: 0.2273 - accuracy: 0.9073
Evaluating model for 1 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 0.1889 - accuracy: 0.9277
[0.18893255293369293, 0.9277166724205017]



26th iteration, adding two more layers for the lols

Fitting data for 1 epochs
12283/12283 [==============================] - 168s 14ms/step - loss: 0.2319 - accuracy: 0.9058
Evaluating model for 1 epochs
4079/4079 [==============================] - 18s 4ms/step - loss: 0.1861 - accuracy: 0.9280
[0.1861400008201599, 0.9279618263244629]



27th iteration, testing the loss-function kld


Fitting data for 1 epochs
12283/12283 [==============================] - 147s 12ms/step - loss: 0.0126 - accuracy: 0.7032
Evaluating model for 1 epochs
4079/4079 [==============================] - 16s 4ms/step - loss: 1.1127e-05 - accuracy: 0.7061
[1.1127447578473948e-05, 0.7061243653297424]



28th iteration, testing MSE as loss funciton

Fitting data for 1 epochs
12283/12283 [==============================] - 163s 13ms/step - loss: 0.0720 - accuracy: 0.9034
Evaluating model for 1 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 0.0543 - accuracy: 0.9280
[0.05425582453608513, 0.928030788898468]

Really nice loss here! Good for discussion (although it is way to many things to discuss...)
But still nice to get a really low loss for almost no cost whatsofucking ever



29th iteration, testing binary_focal_crossentropy

Fitting data for 1 epochs
12283/12283 [==============================] - 161s 13ms/step - loss: 0.0695 - accuracy: 0.8955
Evaluating model for 1 epochs
4079/4079 [==============================] - 18s 4ms/step - loss: 0.0558 - accuracy: 0.9118
[0.055794209241867065, 0.9117966890335083]


30th iteration, testing huber loss function

Fitting data for 1 epochs
12283/12283 [==============================] - 155s 12ms/step - loss: 0.0357 - accuracy: 0.9044
Evaluating model for 1 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 0.0276 - accuracy: 0.9248
[0.02763703465461731, 0.9247594475746155]


Question, as the loss for the later iterations have been amazing, how does that affect any overfitting?
As the loss is getting down towards really low numbers, does this mean that the data is slightly 
overfitted. Running the data with 10 iterations to check the effect

Fitting data for 10 epochs
Epoch 1/10
12283/12283 [==============================] - 146s 12ms/step - loss: 0.0362 - accuracy: 0.9029
Epoch 2/10
12283/12283 [==============================] - 146s 12ms/step - loss: 0.0258 - accuracy: 0.9315
Epoch 3/10
12283/12283 [==============================] - 147s 12ms/step - loss: 0.0229 - accuracy: 0.9391
Epoch 4/10
12283/12283 [==============================] - 148s 12ms/step - loss: 0.0211 - accuracy: 0.9443
Epoch 5/10
12283/12283 [==============================] - 148s 12ms/step - loss: 0.0199 - accuracy: 0.9482
Epoch 6/10
12283/12283 [==============================] - 148s 12ms/step - loss: 0.0187 - accuracy: 0.9515
Epoch 7/10
12283/12283 [==============================] - 149s 12ms/step - loss: 0.0178 - accuracy: 0.9541
Epoch 8/10
12283/12283 [==============================] - 149s 12ms/step - loss: 0.0169 - accuracy: 0.9567
Epoch 9/10
12283/12283 [==============================] - 149s 12ms/step - loss: 0.0162 - accuracy: 0.9586
Epoch 10/10
12283/12283 [==============================] - 149s 12ms/step - loss: 0.0155 - accuracy: 0.9608
Evaluating model for 10 epochs
4079/4079 [==============================] - 17s 4ms/step - loss: 0.0214 - accuracy: 0.9449
[0.021404903382062912, 0.9449236989021301]


"""

"""
Questions regarding the performance of the algorithm:
  - why is the loss constantly 0.0000?
    * stupid me! I should be shot for not understanding this earlier
  - how is the performance affected by the different layers?
  - what is the optimal way to create the different layers to achieve optimal
      performance?
  - which parameters would be required to tune to get an improved performance? 
"""



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

    # Exception: np.float64 cannot be interpreted as an integer
    # Must create x-data to be a list of integers! 
    # Error caused by the max-len not being an integer
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

    # Somehow keras.utils does not contain to_categorical, despite
    # https://www.tensorflow.org/api_docs/python/tf/keras/utils/to_categorical
    # self.__y_train = utils.to_categorical(raw_y_train, 1, dtype=int)
    # self.__y_test = utils.to_catecorical(raw_y_test, 1, dtype=int)

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
    output_embedded_dim = 32
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

    # # Trying with another LSTM layer - didn't work very well
    # model.add(
    #   layers.LSTM(
    #     units=lstm_units
    #   )
    # )

    # Dense layer will implement an activation + bias function.
    # https://keras.io/api/layers/core_layers/dense/

    # # Extra layers added for science
    # model.add(
    #   layers.Dense(
    #     units=first_dense_units,
    #     activation="relu" 
    #   )
    # )

    # model.add(
    #   layers.Dense(
    #     units=second_dense_units,
    #     activation="relu" 
    #   )
    # )

    # model.add(
    #   layers.Dense(
    #     units=third_dense_units,
    #     activation="relu" 
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
      batch_size=None, # Do not specify the batch_size if your data is in the form of datasets, generators, or keras.utils.Sequence instances (since they generate batches). 
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
