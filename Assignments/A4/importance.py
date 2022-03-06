import numpy as np

import random

class Importance:
  def random(
      self,
      attribute_array : np.ndarray
    ) -> int:
    """
    Randomly selects an attribute to separate on  
    """
    attribute_array = attribute_array.reshape((1,-1))
    assert attribute_array.shape[0] > 0 and attribute_array.shape[1] > 0
    return random.randint(0, attribute_array.shape[1] - 1)

  def expected_information(
      self,
      data_node_list : list
    ) -> int:
    """
    Using the information gain from a given attribute, to allocate the importance
    function for said attribute 
    """
    
    return 0

  def remainder(self):
    return 0


  def information_gain(self):
    return 0

  


