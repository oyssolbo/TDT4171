from cProfile import label
import os 
import sys 

import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn import feature_extraction, naive_bayes, tree
from sklearn.metrics import accuracy_score

class MLScikit:
  def __init__(
        self, 
        filename      : str = 'data/scikit-learn-data.pickle'
      ) -> None:
    self.__filename = os.path.join(sys.path[0], filename)
    self.__is_initialized = False

    self.__data = self.__unpickle_data()

  def initialize_data(
        self,
        num_features : int = 14
      ) -> None:
    """
    Initializes the training and test-data into arrays that could 
    be trained and tested on
    """

    raw_x_train = self.__data["x_train"]
    raw_y_train = self.__data["y_train"]

    raw_x_test = self.__data["x_test"]
    raw_y_test = self.__data["y_test"]

    stop_words = self.__get_stop_words()

    vectorizer = feature_extraction.text.HashingVectorizer(
      stop_words=stop_words,
      n_features=2**num_features,
      binary=True
    )

    self.__x_train = vectorizer.transform(X=raw_x_train)
    self.__y_train = raw_y_train 

    self.__x_test = vectorizer.transform(X=raw_x_test)
    self.__y_test = raw_y_test 

    self.__is_initialized = True

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

  def __get_stop_words(self) -> list:
    """
    Returns a list of common stop-words for the English language

    Warning: The function might have a problem regarding 
    words containing '. For example "haven't" will according to 
    https://scikit-learn.org/stable/modules/feature_extraction.html#stop-words
    be split into haven and t. As of 20.03.22, nothing else is implemented
    for this
    """
    stop_words = list(set(stopwords.words("english")))
    return stop_words

  def __train_classifier(
        self,
        classifier
      ) -> tuple:
    """
    Trains and tests for a random classifier. Returns normnalized 
    values indicating how well the training and the testing performed

    Returns a tuple containing:
      -training_accuracy : Results from the training-set. Normalized to range [0, 1]
      -testing_accuracy  : Results from the testing-set. Normalized to range [0, 1]
    """
    assert isinstance(classifier, naive_bayes.BernoulliNB) or isinstance(classifier, tree.DecisionTreeClassifier), "Classifier not supported"

    classifier.fit(self.__x_train, self.__y_train)

    y_train_pred = classifier.predict(X=self.__x_train)
    y_test_pred = classifier.predict(X=self.__x_test)

    training_accuracy = accuracy_score(
      y_true=self.__y_train, 
      y_pred=y_train_pred, 
      normalize=True
    )
    testing_accuracy = accuracy_score(
      y_true=self.__y_test, 
      y_pred=y_test_pred, 
      normalize=True
    )

    return (training_accuracy, testing_accuracy)

  def naive_bayes(self, *args) -> tuple:
    """
    Trains and tests a naive bayes model, using the function __train_classifier(). 
    Returns normnalized values indicating how well the training and the testing performed

    Returns a tuple containing:
      -training_accuracy : Results from the training-set. Normalized to range [0, 1]
      -testing_accuracy  : Results from the testing-set. Normalized to range [0, 1]
    """    
    if not self.__is_initialized:
      warnings.warn("Data not initialized. Defaulting to 14 parameters")
      self.initialize_data()

    # Unsure regarding the parameters - wanted to take these in as parameters
    classifier = naive_bayes.BernoulliNB() 
    return self.__train_classifier(classifier=classifier)

  def decision_tree(
        self, 
        tree_depth : int = 14, 
        *args
      ) -> tuple:
    """
    Trains and tests a decision tree. Returns normnalized 
    values indicating how well the training and the testing performed

    Returns a tuple containing:
      -training_accuracy : Results from the training-set. Normalized to range [0, 1]
      -testing_accuracy  : Results from the testing-set. Normalized to range [0, 1]
    """
    if not self.__is_initialized:
      warnings.warn("Data not initialized. Defaulting to 14 parameters")
      self.initialize_data()

    # Unsure regarding the parameters - wanted to take these in as parameters, such
    # that one could iteratively run different parameters
    classifier = tree.DecisionTreeClassifier(max_depth=tree_depth) 
    return self.__train_classifier(classifier=classifier)

def plot_accuracies(
      training_results  : np.ndarray, 
      testing_results   : np.ndarray, 
      feature_array     : np.ndarray,
      classifier_str    : str
    ) -> None:
  """
  Creates a scatter plot comparing the accuracies of the training vs test-data
  for the different num_features  
  """
  plt.title("Accuracy of training and testing using method {}".format(classifier_str))
  plt.scatter(x=feature_array, y=training_results, c='r', label='train')
  plt.scatter(x=feature_array, y=testing_results, c='g', label='test')
  plt.legend(loc='lower right')
  plt.show()

if __name__ == '__main__':
  num_features_list = list(range(2, 4, 2))
  tree_depths_list = list(range(2, 22, 2))

  test_tree_depth = False

  # [training, testing]^T
  nb_results = np.zeros((2, len(num_features_list)))
  if not test_tree_depth:
    dt_results = np.zeros((2, len(num_features_list)))
  else:
    dt_results = np.zeros((2, len(tree_depths_list)))

  # naive_bayes_args = {}
  # decision_tree_args = {'max_depth': 14}

  ml_scikit = MLScikit()

  for (idx, num_features) in enumerate(num_features_list):
    print("Running n_features = {}".format(num_features))
    
    ml_scikit.initialize_data(num_features=num_features)

    if not test_tree_depth:
      # This is the standard method, which by default runs the decision tree
      # (unless otherwise specified) with 14 as a maximum dep

      print("Naive bayes for n_features = {}".format(num_features))
      (nb_training_results, nb_testing_results) = ml_scikit.naive_bayes()

      print("Decision tree for n_features = {}".format(num_features))
      (dt_training_results, dt_testing_results) = ml_scikit.decision_tree()

      nb_results[:, idx] = np.array([nb_training_results, nb_testing_results]).T
      dt_results[:, idx] = np.array([dt_training_results, dt_testing_results]).T

    else:
      assert len(num_features_list) == 1, "Code not set up for larger arrays"
      for (depth_idx, depth) in enumerate(tree_depths_list):
        print("Decision tree for depth = {}".format(depth))
        (dt_training_results, dt_testing_results) = ml_scikit.decision_tree(tree_depth=depth)#None) # None for testing maximum depth
        dt_results[:, depth_idx] = np.array([dt_training_results, dt_testing_results]).T
        break
    
    # Temporary, such that one is not required to run through all iterations
    # if idx >= 0:
    #   break

  np.savetxt(os.path.join(sys.path[0], "results/data/nb_results.txt"), nb_results)
  np.savetxt(os.path.join(sys.path[0], "results/data/dt_results.txt"), dt_results)

  nb_training_results = nb_results[0,:]
  nb_testing_results = nb_results[1,:]

  dt_training_results = dt_results[0,:]
  dt_testing_results = dt_results[1,:]

  print(
    "Normalized accuracies obtained for scikit naive bayesian: \n \
    training_results: {} \n \
    testing_results: {}".format(
      nb_training_results,
      nb_testing_results
    )
  )

  print(
    "Normalized accuracies obtained for scikit decision tree: \n \
    training_results: {} \n \
    testing_results: {}".format(
      dt_training_results,
      dt_testing_results
    )
  )

  plot_accuracies(
    training_results=nb_training_results, 
    testing_results=nb_testing_results,
    feature_array=np.array(num_features_list),
    classifier_str="scikit naive bayes"
  )

  if not test_tree_depth:
    feature_list = num_features_list
  else:
    if None in tree_depths_list:
      # Set None-values to -1.0 to prevent errors during plotting
      for (i, d) in enumerate(tree_depths_list):
        tree_depths_list[i] = -1.0
    feature_list = tree_depths_list


  plot_accuracies(
    training_results=dt_training_results, 
    testing_results=dt_testing_results,
    feature_array=np.array(feature_list),
    classifier_str="scikit decision tree"
  )

