import numpy as np
import nodes
import random
import warnings

class Importance:
  # def random(
  #     self,
  #     attribute_array : np.ndarray
  #   ) -> int:
  #   """
  #   Randomly selects an attribute to separate on  
  #   """
  #   attribute_array = attribute_array.reshape((1,-1))
  #   assert attribute_array.shape[0] > 0 and attribute_array.shape[1] > 0
  #   return random.randint(0, attribute_array.shape[1] - 1)

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
      if probability <= 1e-12: return 0
      return -(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability))

    # Extracting total number of 'positive' and 'negative'    
    # Using prior knowledge that the values will only be 1 or 2
    min_val = 1
    max_val = 2

    num_min_vals = 0
    num_max_vals = 0

    for node in data_node_list:
      data_type = node.get_type()
      if data_type == min_val:
        num_min_vals += 1
      elif data_type == max_val:
        num_max_vals += 1
      else:
        warnings.warn("Other value detected: {}".format(data_type))

    # Iterating through all nodes, and determining the optimal attribute
    optimal_attribute = 0
    optimal_attribute_num_pos = 0
    # for node in 
    # for val in range(min_val, max_val + 1):

    # Understand how to caculate the remainder

    # Calculate the information-gain

    # Select the attribute which maximizes the information gain

    return 0

  def remainder(self):
    return 0


  def information_gain(self):
    return 0

  


