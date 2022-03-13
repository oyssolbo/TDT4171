import os
import sys

import numpy as np
import graphviz
import csv
import warnings
import copy
import collections
import random

import matplotlib.pyplot as plt
import math
import time

from typing import Callable, TextIO

class DataNode:
  def __init__(
        self, 
        data      : np.ndarray,
        node_type : int,
        children  : list    = [],
        attribute : int     = None
      ) -> None:
    self.__data = data
    self.__node_type = node_type
    self.__children = children
    self.__attribute = attribute

    if node_type is not None and attribute is not None:
      warnings.warn("Incorrect combination of attribute and type set")

  def get_type(self) -> int:
    return self.__node_type

  def get_attribute(self) -> int: 
    return self.__attribute

  def get_data(self) -> np.ndarray:
    return self.__data

  def set_data(
        self, 
        data : np.ndarray
      ) -> None:
    self.__data = data

  def add_child(
        self, 
        child : 'DataNode',
        label : int
      ) -> None:
    self.__children.append((child, label))

  def get_children(self) -> list:
    return self.__children
    

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
    ) -> DataNode:
    """
    Randomly selects an attribute to separate on  
    """
    assert len(data_node_list) > 0
    assert data_node_list[0].get_data().reshape(1,-1).shape[1] > 0
    return random.randint(0, (data_node_list[0].get_data().reshape((1,-1))).shape[1] - 1)

  def expected_information(
      self,
      data_node_list : list#[DataNode]
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






class DecisionTree:
  def __init__(
        self,
        train_data_csv  : TextIO,
        test_data_csv   : TextIO
      ) -> None:

    self.__training_nodes = self.extract_data(train_data_csv)
    self.__testing_nodes = self.extract_data(test_data_csv)

    self.__root_node = None

    # Using prior knowledge that the minimum data is 1 and the maximum
    # data is 2
    self.__min_val = 1
    self.__max_val = 2

    # Showing the depth of the tree
    self.__tree_depth = 0

    # Keeping track of which attribute has been removed. Otherwise, the 
    self.__num_training_attributes = self.__training_nodes[0].get_data().shape[0]
    self.__training_attributes_list = [*range(0, self.__num_training_attributes, 1)]


  def train_decision_tree(
        self,
        importance_func : Callable
      ) -> None:
    """
    Interface for training the decision tree 
    """

    def plurarity_node(
          data_node_list : list#[DataNode]
        ) -> DataNode:
      """
      Extracts the plurarity-value of a set of examples, and determines
      the most common class/node based on said examples
      """
      if not data_node_list:
        data_node = DataNode(
          data=np.array([]), 
          node_type=None,
          children=[]
        )
      else:
        # Create an array of the different classes
        type_arr = np.zeros(len(data_node_list), dtype=int).reshape((1, -1))
        for (idx, node) in enumerate(data_node_list):
          type_arr[0, idx] = node.get_type()

        # Find the type that occurs most often
        most_common_type = collections.Counter(type_arr[0]).most_common(1)[0][0]

        # Create a node with correct type
        data_node = DataNode(
          data=-np.ones_like(data_node_list[0].get_data()), # -1 to represent plurality_node
          node_type=most_common_type,
          children=[]
        )
      return data_node

    def check_classification(
          data_node_list : list = [] #[DataNode] = []
        ) -> tuple: #[bool, DataNode]:
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

      # Checking if all elements are equal
      if not np.all(class_arr == class_arr[0,0]):
        # print(class_arr)
        return (False, None) # Last argument contains no information in this case

      # Create a node with correct type, since all have the same type
      data_node = DataNode(
        data=-2*np.ones_like(data_node_list[0].get_data()), # -2 to represent check_classification
        node_type=class_arr[0,0],
        children=[]
      )
      return (True, data_node)  
      
    def learn_decision_tree(
          importance_func         : Callable,
          current_data_node_list  : list      = [],
          prev_data_node_list     : list      = [],
          available_attributes    : list      = [],
          tree_depth              : int       = 0
        ) -> DataNode:
      """
      Implements the decision-tree algorithm with pseudocode shown in 
      figure 19.5 in Russel and Norvig
      """
      tree_depth += 1
      if tree_depth >= self.__tree_depth:
        self.__tree_depth = tree_depth 

      (has_common_class, common_node) = check_classification(current_data_node_list)

      if not current_data_node_list:
        # examples-list is empty
        # print("not current_data_node_list")
        return plurarity_node(prev_data_node_list)
      elif has_common_class:
        # print("has_common_class")
        return common_node
      elif (
        current_data_node_list[0].get_data().reshape((1, -1)).shape[1] == 0 or \
        len(available_attributes) == 0
      ):
        # attributes-list is empty
        # print("not current_data_node_list[0].get_data()")
        return plurarity_node(current_data_node_list)
      else:
        # Calculate the node with the best attribute
        # The local attribute will only store the attribute observed locally, however
        # we are interested in the global attribute, as the global attributes are 
        # used to separate the nodes from each other. Otherwise, one will get too many
        # syclic nodes in the resulting tree, as all local attributes will converge 
        # onto 0, 1, 2, ... 
        local_attribute = importance_func(current_data_node_list)
        global_attribute = available_attributes[local_attribute]
        available_attributes.remove(global_attribute) 

        root_node = DataNode(
          data=-3*np.ones_like(current_data_node_list[0].get_data()), # -3 to represent else-block
          node_type=None,
          children=[],
          attribute=global_attribute 
        )

        vals = range(self.__min_val, self.__max_val + 1)

        for val in vals: 
          next_node_list = []
          for node in current_data_node_list:
            # Extract the nodes that match 
            node_data = node.get_data()
            if node_data.size == 0:
              warnings.warn("Node contains no data!")
              continue
            if node_data[local_attribute] == val:
              # Add to array for further invokations
              next_node_list.append(copy.copy(node))

          # Removing the attribute for all of the selected nodes
          for (idx, node) in enumerate(next_node_list):
            updated_data = np.delete(node.get_data(), local_attribute)
            next_node_list[idx].set_data(updated_data)

          # Next set of recursion
          root_node.add_child(
            child=learn_decision_tree(
              importance_func=importance_func,
              current_data_node_list=copy.copy(next_node_list),
              prev_data_node_list=copy.copy(current_data_node_list),
              available_attributes=copy.copy(available_attributes),
              tree_depth=tree_depth
            ),
            label=val
          ) 

        # Check that all non-leaf-nodes have correct number of children
        num_children = len(root_node.get_children())
        assert num_children == self.__max_val - self.__min_val + 1, "Internal nodes must have two children"

        # If original root, save it
        # Otherwise return the root node
        if tree_depth == 1:
          self.__root_node = root_node
        return root_node
    
    learn_decision_tree(
      importance_func=importance_func,
      current_data_node_list=self.__training_nodes, 
      prev_data_node_list=[],
      available_attributes=copy.copy(self.__training_attributes_list)
    )


  def document_tree(
        self,
        root_node : DataNode  = None,
        comment   : str             = "Decision tree",
        id        : int             = 0,
        save_tree : bool            = True,
        show_tree : bool            = False
      ) -> None:

    def get_node_name(
          node      : DataNode,
          num_types : list           
        ) -> tuple:#[str, list]:
      data_type = node.get_type()
      attribute = node.get_attribute()

      # Check for invalid combination
      # If both the attribute not None and data_type not None, one cannot
      # know whether it is an internal node or a leaf-node
      if data_type is not None and attribute is not None:
        raise ValueError("Cannot document a tree with current_data_type and current_attribute both being defined")
      
      if data_type is not None:
        # Leaf node. Should not contain children
        # Here there is a bug when creating the  Multiple leaf-nodes are given the same name,
        # and as such, the algorithm is unable to generate multiple leaf nodes, as it cannot know which
        # to refer to. Must add a
        node_label = str(data_type)
        node_name = node_label + "_" + str(num_types[int(data_type) - 1])
        num_types[int(data_type) - 1] += 1 
      else:
        # Internal node. Should contain children
        node_name = "A" + str(attribute)
        node_label = node_name
      return (node_name, node_label, num_types) #(node_name, node_label)


    def build_documented_tree(
        tree            : graphviz.Digraph, 
        current_node    : DataNode, 
        parent_node     : DataNode    = None,
        accounted_nodes : list              = [],
        num_types       : list              = [],
        label           : str               = ""
      ) -> graphviz.Digraph:

      """
      There is something buggy about this function! It will not create correct trees, despite having 
      correct information. Some problems with the function, are:
        -will create syclic trees despite the nodes are not syclic
        -will not add all of the leaf-nodes
      """
      if not num_types:
        num_types = [0] * (self.__max_val - self.__min_val + 1) 

      # Create node
      (current_node_name, current_node_label, num_types) = get_node_name(node=current_node, num_types=num_types)
      current_node_type = current_node.get_type() 
      tree.node(name=current_node_name, label=current_node_label)

      # Add edges to any potential parents
      if parent_node is not None:
        # Parent is an internal node
        (parent_node_name, _, _) = get_node_name(node=parent_node, num_types=[0] * (self.__max_val - self.__min_val + 1))

        tree.edge(tail_name=parent_node_name, head_name=current_node_name, label=label)

      if current_node_type is None:
        # This test (atleast when I am testing it), is never triggered
        # That means that all nodes contains two children each
        assert len(current_node.get_children()) == 2, "Internal node must have two children"      

        # Bad debugging code for testing that the values are different - they are different
        # children = current_node.get_children()
        # temp = None
        # for (child, val) in children:
        #   print(val)
        #   if temp is None:
        #     temp = val 
        #     continue
        #   assert val != temp, "Must be different"

      # Something buggy here! 
      # Tried to iterate through all of the children, but only if they are not accounted for earlier
      # The thougth was to prevent syclic behaviour if a node has multiple parents
      # recursed_nodes = []
      for (child_node, val) in current_node.get_children():
        (child_node_name, _, num_types) = get_node_name(node=child_node, num_types=num_types)
        if child_node_name in accounted_nodes: 
          # Prevent cyclic behaviour
          continue
        
        child_node_type = child_node.get_type()
        if child_node_type is None:
          # Only add the attributes
          accounted_nodes.append(child_node_name)
        
        build_documented_tree(
          tree=tree, 
          current_node=child_node, 
          parent_node=current_node,
          accounted_nodes=accounted_nodes, 
          num_types=[0] * (self.__max_val - self.__min_val + 1),
          label=str(val)
        )         

      return tree

    def save_tree(
        tree : graphviz.Digraph,
        show : bool             = False 
      ) -> None:
      tree.render(view=show)

    if root_node is None and self.__root_node is None:
      raise ValueError("No node found to document the tree")

    if root_node is None:
      root_node = self.__root_node

    # This method tries to document the tree using graphviz
    tree = graphviz.Digraph(
      comment=comment, 
      filename=os.path.join(sys.path[0], "data/results/data/tree/{} id={}.gv".format(comment, id)),
      strict=True)
    tree = build_documented_tree(
      tree=tree, 
      current_node=root_node, 
      parent_node=None,
      accounted_nodes=[]
    )
    if not save_tree:
      return
    if sys.platform.startswith('win32'):
      # The graphical part of graphwiz does somehow not work for me on windows
      return
    save_tree(tree=tree, show=show_tree)

  def test_decision_tree(
        self
      ) -> None:
    """
    Testing how well the trained tree actually is.
    Returns a score of how accurate the algorithm has been into learning
    the system
    """

    def get_matching_leaf_node(
          root_node : DataNode,
          test_node : DataNode
        ) -> DataNode:
      # Tries to get the leaf-node that matches the test-node

      # Copy to prevent stupid python-errors that have plagued my project
      tree_node = copy.copy(root_node)
      node_data = test_node.get_data()

      while tree_node.get_attribute() is not None:
        attribute = tree_node.get_attribute()
        children = tree_node.get_children()

        found_value = False 

        # Find which child matches
        for (child, val) in children:
          if node_data[attribute] == val:
            found_value = True 
            tree_node = child
            break

        if not found_value:
          # No child supporting the test was found
          return None

      return copy.copy(tree_node)
      

    test_nodes = self.__testing_nodes
    num_correct = 0

    # Iterate through all of the nodes, and test if they are classified correctly
    for test_node in test_nodes:
      # Start from the root for all test-nodes
      # Get a leaf node 
      leaf_node = get_matching_leaf_node(root_node=self.__root_node, test_node=test_node)

      if leaf_node is None:
        # No matching leaf node found
        continue
      
      # Once a leaf-node is found, we must test if the detected value is correct
      tree_type = leaf_node.get_type()
      if test_node.get_type() == tree_type:
        num_correct += 1

    # print("Number correct {}, of {} possible. Proportion correct on test-set: {}".format(num_correct, len(test_nodes), num_correct / len(test_nodes)))
    return num_correct

  def extract_data(
        self,
        data_csv : TextIO,
      ) -> list:#[DataNode]:
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
        node = DataNode(data[:-1], node_type)
        data_node_list.append(node)
    return data_node_list

if __name__ == '__main__':
  train_data = os.path.join(sys.path[0], 'data/train.csv')
  test_data = os.path.join(sys.path[0], 'data/test.csv')

  importance_class = Importance(min_val=1, max_val=2)
  random_importance_func = lambda x : importance_class.random(x)
  expected_information_importance_func = lambda x : importance_class.expected_information(x)

  num_tests = 500 #5000
  random_correct_arr = np.zeros(num_tests, dtype=int)
  expected_correct_arr = np.zeros(num_tests, dtype=int)

  random_cpu_count_arr = np.zeros(num_tests, dtype=int)
  expected_cpu_count_arr = np.zeros(num_tests, dtype=int)

  # Random tree
  random_tree = DecisionTree(
    train_data_csv=train_data, 
    test_data_csv=test_data
  )

  # Expected information tree
  expected_information_tree = DecisionTree(
    train_data_csv=train_data, 
    test_data_csv=test_data
  )

  # Iterating over the algorithm for n iterations to create some data
  for i in range(num_tests):
    cpu_count = time.perf_counter_ns()
    random_tree.train_decision_tree(importance_func=random_importance_func)
    random_tree.document_tree(
      root_node=None, 
      comment="Random importance",
      id=i,
      save_tree=False,
      show_tree=True
    )
    random_result = random_tree.test_decision_tree()
    random_count = time.perf_counter_ns() - cpu_count

    cpu_count = time.perf_counter_ns()
    expected_information_tree.train_decision_tree(importance_func=expected_information_importance_func)
    expected_information_tree.document_tree(
      root_node=None, 
      comment="Expected information importance",
      id=i,
      save_tree=False,
      show_tree=True
    )
    expected_result = expected_information_tree.test_decision_tree()
    expected_count = time.perf_counter_ns() - cpu_count

    # Save data temporary
    random_correct_arr[i] = int(random_result)
    expected_correct_arr[i] = int(expected_result)

    random_cpu_count_arr[i] = random_count
    expected_cpu_count_arr[i] = expected_count

    if i % 100 == 0 and i > 0:
      print(i / num_tests)

  # Save data permanently
  np.savetxt(os.path.join(sys.path[0], "data/results/data/random_results.txt"), random_correct_arr)
  np.savetxt(os.path.join(sys.path[0], "data/results/data/expected_results.txt"), expected_correct_arr)

  # Calculate some statistics
  random_correct_mean = np.mean(random_correct_arr)
  random_correct_var = np.var(random_correct_arr)

  random_cpu_count_mean = np.mean(random_cpu_count_arr)
  random_cpu_count_var = np.var(random_cpu_count_arr)

  expected_correct_mean = np.mean(expected_correct_arr)
  expected_correct_var = np.var(expected_correct_arr)

  expected_cpu_count_mean = np.mean(expected_cpu_count_arr)
  expected_cpu_count_var = np.var(expected_cpu_count_arr)

  # Find percentile of times where the expected is better than the random
  num_random_similar = len(random_correct_arr[random_correct_arr == expected_correct_mean])
  num_random_better = len(random_correct_arr[random_correct_arr > expected_correct_mean])

  # Plot histograms
  fig, axs = plt.subplots(2)
  fig.suptitle(
    "Histogram comparison for {} tests.\n Random importance function similar in {} test(s) and better in {} test(s)".format(
      num_tests,
      num_random_similar,
      num_random_better
    )
  )

  axs[0].set_title(
    "Random importance function with mean: {} and variance: {}".format(
      '%.1f'%random_correct_mean, 
      '%.1f'%random_correct_var
    )
  )
  axs[0].hist(random_correct_arr, bins=np.arange(12,29,1))

  axs[1].set_title(
    "Expected information importance function with mean: {} and variance: {}".format(
      '%.1f'%expected_correct_mean, 
      '%.1f'%expected_correct_var
    )
  )
  axs[1].hist(expected_correct_arr, bins=np.arange(12,29,1))

  print(
    "Random cpu-count. Mean: {} Variance: {}".format(
      random_cpu_count_mean, 
      random_cpu_count_var
    )
  )

  print(
    "Expected cpu-count. Mean: {} Variance: {}".format(
      expected_cpu_count_mean, 
      expected_cpu_count_var
    )
  )

  plt.show()
