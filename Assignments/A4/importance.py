import numpy as np
import nodes
import random
import warnings

class Importance:
  def __init__(
        self,
        min_val : int,
        max_val : int
      ) -> None:
    self.min_val = min_val
    self.max_val = max_val

  def random(
      self,
      data_node_list : list
    ) -> nodes.DataNode:
    """
    Randomly selects an attribute to separate on  
    """
    assert len(data_node_list) > 0
    assert data_node_list[0].get_data().reshape(1,-1).shape[1] > 0
    return random.randint(0, (data_node_list[0].get_data().reshape((1,-1))).shape[1] - 1)

  def expected_information(
      self,
      data_node_list : list
    ) -> int:
    """
    Using the information gain from a given attribute, to allocate the importance
    function for said attribute 

    The function must take in or extract the numer of positive and number of negative
    values in the test-set
    """

    def binary_entropy(probability : float) -> float:
      assert probability >= 0 and probability <= 1, "Probability out of range"
      if probability <= 1e-12: return 0 # To prevent problems with lb(0)
      return -(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability))

    assert isinstance(data_node_list, list), "Training data must be given as a list"
    assert len(data_node_list) > 0, "No data received"

    # Iterate through all of the columns and extract number of 1 and 2 for each columnn
    num_vals_per_column = np.zeros((self.max_val - self.min_val + 1, data_node_list[0].get_data().reshape((1,-1)).shape[1]), dtype=int)

    for node in data_node_list:
      node_data = node.get_data()
      for (attribute, data) in enumerate(node_data):
        for val in range(self.min_val, self.max_val + 1):
          if val == data:
            num_vals_per_column[val - self.min_val, attribute] += 1

    # Finding total amount of values
    # total_num_min_vals = np.sum(num_vals_per_column[0])
    # total_num_max_vals = np.sum(num_vals_per_column[1])

    # Calculating total binary entropy
    # total_bin_entropy = binary_entropy(probability=(num_vals_per_column[self.min_val] / len(data_node_list)))

    #binary_entropy(probability=(total_num_min_vals / (total_num_min_vals + total_num_max_vals)))

    # Using entropy to calculate the optimal entropy
    gain_arr = -np.ones((1, num_vals_per_column.shape[1]), dtype=float).reshape((1,-1))

    for attribute in range(num_vals_per_column.shape[1]):
      # Calculate binary entropy for a given attribute
      bin_entropy = binary_entropy(probability=(num_vals_per_column[self.min_val, attribute] / len(data_node_list)))

      # Calculate information gain
      num_mins = num_vals_per_column[0, attribute]
      num_max = num_vals_per_column[1, attribute]

      gain = 0

      # Thinks that the problem is here somewhere. As written in the book:
      # Each subset Ek has pk positive examples and nk negative examples. Would this mean that these lines must be changed?
      attribute_probability = (num_mins + num_max) / len(data_node_list) #(total_num_min_vals + total_num_max_vals)
      binary_probability = num_mins / (num_mins + num_max)
      gain_arr[0, attribute] = bin_entropy - (attribute_probability * binary_entropy(probability=binary_probability))

    # Calculating the information gain. Return attribute which maxes this
    # gain_arr = 0# total_bin_entropy - remainder_arr
    print(np.argmax(gain_arr))

    return np.argmax(gain_arr)
