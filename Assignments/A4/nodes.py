import numpy as np

from numpy import ndarray
from typing import Callable

class DataNode:
  def __init__(
        self, 
        data      : ndarray,
        node_type : int,
        children  : list    = []
      ) -> None:
    self.data = data
    self.node_type = node_type
    self.children = children

  def get_type(self) -> int:
    return self.node_type

  def get_data(self) -> ndarray:
    return self.data

  def add_child(
        self, 
        child : 'DataNode',
        label : int
      ) -> None:
    self.children.append((child, label))

# class AttributeNode:
#   def __init__(
#         self,
#         parent_node         = None,
#         children    : list  = [],

#       ) -> None:
#     pass