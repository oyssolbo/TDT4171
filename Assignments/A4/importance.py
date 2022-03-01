import numpy as np
import nodes

import random

class Importance:
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

  def expected_information(self):
    """
    Using the information gain from a given attribute, to allocate the importance
    function for said attribute 
    """
    
    return 0

  def remainder(self):
    return 0


  def information_gain(self):
    return 0

  


