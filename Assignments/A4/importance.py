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

    if self.max_val - self.min_val == 1:
      self.pos_idx = self.max_val
      self.neg_idx = self.min_val
    else:
      raise ValueError("No identifier found for positive / negative idx")

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
      data_node_list : list[nodes.DataNode]
    ) -> int:
    """
    Using the information gain from a given attribute, to allocate the importance
    function for said attribute 

    The function must take in or extract the numer of positive and number of negative
    values in the test-set
    """

    def binary_entropy(probability : float) -> float:
      assert isinstance(probability, float), "Probability not given as a float"
      assert probability >= 0 and probability <= 1, "Probability out of range"
      if probability <= 1e-8 or probability >= (1 - 1e-8): return 0 # To prevent problems with lb(0)
      return -(probability * np.log2(probability) + (1 - probability) * np.log2(1 - probability))

    assert isinstance(data_node_list, list), "Training data must be given as a list"
    assert len(data_node_list) > 0, "No data received"

    # Iterate through all of the nodes, and extract data
    num_attributes = data_node_list[0].get_data().reshape((1,-1)).shape[1]
    num_values = self.max_val - self.min_val + 1
    assert num_values > 0, "Too small range to calculate expected information"

    total_attribute_arr = np.zeros((len(data_node_list), num_attributes), dtype=int)
    attribute_match_arr = np.zeros((2 * num_values, num_attributes), dtype=int) # Num pos and num neg for each attribute and identifier
    type_arr = np.zeros(len(data_node_list), dtype=int)

    for (idx, node) in enumerate(data_node_list):
      node_data = node.get_data()
      node_type = node.get_type()

      total_attribute_arr[idx, :] = node_data
      type_arr[idx] = node_type

    # Iterate through all of the values in the column and detect number of positive
    # and negative for each attribute
    for col in range(num_attributes):
      attribute_col = total_attribute_arr[:, col]

      for val in range(self.min_val, self.max_val + 1):
        num_pos = 0
        num_neg = 0
        for (idx, a) in enumerate(attribute_col):
          if val == a:
            if type_arr[idx] == self.pos_idx:
              num_pos += 1
            else:
              num_neg += 1
        
        val_idx = val - self.min_val
        attribute_match_arr[num_values * val_idx : num_values * val_idx + num_values, col] = np.array([num_neg, num_pos]).T

    # Finding total number of positive and negative types
    counted_type_arr = np.bincount(type_arr)

    total_num_neg = counted_type_arr[self.neg_idx]
    total_num_pos = counted_type_arr[self.pos_idx]

    # Finding attribute with the best information
    gain_arr = np.zeros((1, num_attributes))

    for col in range(num_attributes):
      remainder = 0
      for val in range(num_values):
        nk = attribute_match_arr[num_values * val, col]   
        pk = attribute_match_arr[num_values * val + (num_values - 1), col] # -1 to create an index

        if pk + nk == 0:
          print(attribute_match_arr)
          raise ValueError("Division by zero")

        entropy = binary_entropy(probability=(pk / (pk + nk)))
        attribute_probability = (pk + nk) / (total_num_pos + total_num_neg)
        remainder += attribute_probability * entropy

      if total_num_pos + total_num_neg == 0:
        raise ValueError("Division by zero")
      gain_arr[0, col] = binary_entropy(probability=(total_num_pos / (total_num_pos + total_num_neg))) - remainder
    
    return np.argmax(gain_arr)


    """
    A prior attempt at doing this task
    """

    attribute_idx = np.argmax(gain_arr)


    print(gain_arr)
    print(attribute_idx)
    # print(total_num_neg)
    quit()














    # correct_classifications = np.zeros((1, len(data_node_list)), dtype=int)
    num_positive_classifier_per_attribute = np.zeros((self.max_val - self.min_val + 1, data_node_list[0].get_data().reshape((1,-1)).shape[1]), dtype=int)

    for (idx, node) in enumerate(data_node_list):
      # True type
      node_type = node.get_type()
      # correct_classifications[idx] = node_type

      # Attribute-data
      node_data = node.get_data()
      for (attribute, _) in enumerate(node_data):
        for val in range(self.min_val, self.max_val + 1):
          if val == node_type:
            num_positive_classifier_per_attribute[val - self.min_val, attribute] += 1

    
    

    # Finding total amount of values
    # total_num_min_vals = np.sum(num_positive_classifier_per_attribute[0])
    # total_num_max_vals = np.sum(num_positive_classifier_per_attribute[1])

    # Calculating total binary entropy
    # total_bin_entropy = binary_entropy(probability=(num_positive_classifier_per_attribute[self.min_val] / len(data_node_list)))

    #binary_entropy(probability=(total_num_min_vals / (total_num_min_vals + total_num_max_vals)))

    # Using entropy to calculate the optimal entropy
    gain_arr = -np.ones((1, num_positive_classifier_per_attribute.shape[1]), dtype=float).reshape((1,-1))

    for attribute in range(num_positive_classifier_per_attribute.shape[1]):
      # Detect number of positive and negative being found with this split

      #





      # Calculate binary entropy for a given attribute
      bin_entropy = binary_entropy(probability=(num_positive_classifier_per_attribute[self.min_val, attribute] / len(data_node_list)))

      # Calculate information gain
      num_mins = num_positive_classifier_per_attribute[0, attribute]
      num_max = num_positive_classifier_per_attribute[1, attribute]

      gain = 0

      # Thinks that the problem is here somewhere. As written in the book:
      # Each subset Ek has pk positive examples and nk negative examples. Would this mean that these lines must be changed?

      # Theory: By the positive and negative values, does that mean the actual training-data? For example if the subset is
      # created based on attribute 0, are the positive and negative values to be calculated as number of true / false in the
      # type of the classes?
      # Based on figure 19.4, so yes!
      # Finally some progress / understanding of the system! This was a thing that bothered me previously, as I couldn't quite
      # figure out the understanding of positive and negative values 

      attribute_probability = (num_mins + num_max) / len(data_node_list) #(total_num_min_vals + total_num_max_vals)
      binary_probability = num_mins / (num_mins + num_max)
      gain_arr[0, attribute] = bin_entropy - (attribute_probability * binary_entropy(probability=binary_probability))

    # Calculating the information gain. Return attribute which maxes this
    # gain_arr = 0# total_bin_entropy - remainder_arr
    print(np.argmax(gain_arr))

    return np.argmax(gain_arr)
