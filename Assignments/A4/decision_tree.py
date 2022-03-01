import os
import sys

import numpy as np
import graphviz
import csv

import importance
import nodes

from typing import Callable, TextIO
from numpy import ndarray

class DecisionTree:
  def __init__(
        self,
        train_data_csv  : TextIO,
        test_data_csv   : TextIO
      ) -> None:

    self.training_nodes = self.extract_data(train_data_csv)
    self.testing_nodes = self.extract_data(test_data_csv)

  def train_decision_tree(
        self,
        importance_func : Callable  
      ) -> None:
    """
    Interface for training the decision tree 
    """

    def plurarity_node(
          data_node_list : list
        ) -> nodes.DataNode:
      """
      Extracts the plurarity-value of a set of examples, and determines
      the most common class/node based on said examples
      """
      if not data_node_list:
        data_node = nodes.DataNode(
          data=np.empty_like(data_node_list[0].get_data()),
          node_type=None,
          children=[]
        )
      else:
        # Create an array of the different classes
        class_arr = np.zeros((1, (len(data_node_list)))).reshape((1, -1))
        for (idx, node) in enumerate(data_node_list):
          class_arr[0, idx] = node.get_type()

        # Find the type that occurs most often
        counts = np.bincount(class_arr)
        most_common_type = np.argmax(counts)

        # Create a node with correct type
        data_node = nodes.DataNode(
          data=np.empty_like(data_node_list[0].get_data()),
          node_type=most_common_type,
          children=[]
        )
      print(data_node)
      return data_node

    def check_classification(
          data_node_list : list = []
        ) -> tuple:
      """
      Checks the classification of the data-nodes, and determines if they 
      all have the same type
      """
      if not data_node_list:
        return (False, None)

      # Create array
      class_arr = np.zeros((1, (len(data_node_list)))).reshape((1, -1))
      for (idx, node) in enumerate(data_node_list):
        class_arr[0, idx] = node.get_type()

      if not np.all(class_arr == class_arr[0]):
        return (False, None) # Last argument contains no information in this case

      # Create a node with correct type
      data_node = nodes.DataNode(
        data=np.empty_like(data_node_list[0].get_data()),
        node_type=class_arr[0],
        children=[]
      )
      return (True, data_node)  
      
    def learn_decision_tree(
          importance_func         : Callable,
          current_data_node_list  : list = [],
          prev_data_node_list     : list = []
        ) -> nodes.DataNode:
      """
      Implements the decision-tree algorithm with pseudocode shown in 
      figure 19.5 in Russel and Norvig
      """
      has_common_class, common_node = check_classification(current_data_node_list)

      if not current_data_node_list:
        # examples-list is empty
        # This is valid, since one removes a node from the data-list for each
        # invokation
        return plurarity_node(prev_data_node_list)
      elif has_common_class:
        return common_node
      elif not current_data_node_list[0].get_data():
        # attributes-list is empty
        return plurarity_node(current_data_node_list)
      else:
        # Calculate the best attribute
        A = importance_func(current_data_node_list)

        # Create a new tree originating from the best node

        # Iterate over all values from the attribute
        for val in A.get_data():
          # Must find out what to do here with the values

          # Must add the detected node as a child to the attribute

          # What to do with the label?
          pass

        # Return the root of the subtree
        return A
    
    learn_decision_tree(
      importance_func=importance_func,
      current_data_node_list=self.testing_nodes, 
      prev_data_node_list=[]
    )


  def test_decision_tree(self) -> None:
    pass

  def extract_data(
        self,
        data_csv : TextIO,
      ) -> list:
    """
    Extracts the data from a csv_file and creates a set of attributes
    that are returned
    """
    data_node_list = []
    with open(data_csv) as csvfile:
      read_line = csv.reader(csvfile)
      for row in read_line:
        data = np.array(list(map(int, row)))
        node_type = data[-1]
        node = nodes.DataNode(data[:-1], node_type)
        data_node_list.append(node)
    return data_node_list

if __name__ == '__main__':
  train_data = os.path.join(sys.path[0], 'data/train.csv')
  test_data = os.path.join(sys.path[0], 'data/test.csv')

  importance_func = importance.Importance.random

  tree = DecisionTree(
    train_data_csv=train_data, 
    test_data_csv=test_data
  )
  tree.train_decision_tree(importance_func=importance_func)



