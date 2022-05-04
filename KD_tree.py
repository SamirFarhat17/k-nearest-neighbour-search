import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import heapq
from scipy.spatial import distance
from tqdm.notebook import tqdm, trange
import copy
import heapq
from scipy import spatial
import os


class kdNode:
  def __init__(self, data_idx, discriminator, lchild=None, rchild=None):
    self.data_idx = data_idx
    self.discriminator = discriminator
    self.lchild = lchild
    self.rchild = rchild
    
  def __repr__(self):
    return f"Node with Index {self.data_idx}"


class knnHeapNode:
  def __init__(self, kdnode, L2Dist):
    self.node = kdnode
    self.L2Dist = L2Dist

  # override the less-than operator to change heapq from min heap to max heap when made of knnHeapNodes
  def __lt__(self, other):
    return self.L2Dist > other.L2Dist

  def __repr__(self):
    return f"(Node Index: {self.node.data_idx}, L2 Distance: {self.L2Dist})"


class kdTree:
  
  def __init__(self):
    self.root = None
    self.data = None

  def builder(self, data, discriminator=0):
    # called by build_tree function

    num_samples = len(data)
    kdim = len(data[0][1])
    median_idx = num_samples // 2
    
    # sort data according to feature column indexed by discriminator
    data = sorted(data, key=lambda x: x[1][discriminator])

    # find first occurrence of median value for discriminator dimension
    median = data[median_idx][1][discriminator]

    while median_idx > 0 and median == data[median_idx-1][1][discriminator]:
      median_idx = median_idx-1

    # create root node for tree
    node = kdNode(data_idx=data[median_idx][0], discriminator=discriminator)

    # calculate discriminator index for next level
    next_discriminator = (discriminator + 1) % kdim

    # add left subtree containing vectors[discriminator] < median and right subtree containing vectors[discriminator] >= median to root
    if median_idx > 0:
      node.lchild = self.builder(data[:median_idx], next_discriminator)
    if num_samples - (median_idx + 1) > 0:
      node.rchild = self.builder(data[median_idx+1:], next_discriminator)

    return node

  def build_tree(self, data):
    # creates kd tree from data
    self.data = copy.deepcopy(data)
    self.root = self.builder(data)

  def find_kNN(self, k, query_vec, node=None, kNN=None):
    
    if node == None:
      node = self.root
    
    # create max heap for tracking node values and associated L2 distances of k nearest neighbors
    if kNN == None:
      kNN = []
      heapq.heapify(kNN)

    # extract feature vector from tuple in reference data
    feat_vec = self.data[node.data_idx][1]

    # calculate L2 distance between query_vec and the current node
    dist = distance.euclidean(query_vec, feat_vec)

    if len(kNN) < k:
      heapq.heappush(kNN, knnHeapNode(node, dist))
    else:
      if dist < kNN[0].L2Dist:
        heapq.heapreplace(kNN, knnHeapNode(node, dist))

    # determine best direction to traverse tree
    discriminator = node.discriminator

    if query_vec[discriminator] < feat_vec[discriminator]:
      good_side = node.lchild
      bad_side = node.rchild
    else:
      good_side = node.rchild
      bad_side = node.lchild

    # traverse good side
    if good_side != None:
      kNN = self.find_kNN(k, query_vec, node=good_side, kNN=kNN)

    # determine if the bad side is worth exploring by checking the possibility of a shorter distance to the query on the bad side
    if bad_side != None and abs(query_vec[discriminator] - feat_vec[discriminator]) < kNN[0].L2Dist:
      kNN = self.find_kNN(k, query_vec, node=bad_side, kNN=kNN)

    # return max heap of k closest nodes
    return kNN


# kd trees test code

class Stack:
  def __init__(self):
    self.items = []
    self.size = 0

  def push(self, item):
    self.items.append(item)
    self.size += 1

  def pop(self):
    self.size -= 1
    return self.items.pop()
  
  def top(self):
    return self.items[self.size-1]

  def isEmpty(self):
    return self.size == 0

# test kd tree creation
sample_data = [(0,[4,6]), (1,[4,5]), (2,[4,2]), (3,[1,4])]

kd_tree = kdTree()
kd_tree.build_tree(sample_data)
root = kd_tree.root

print(kd_tree)
print(root)

# visualize kd-tree using stack

node_stack = Stack()
node_stack.push(root)

while not node_stack.isEmpty():
  current = node_stack.pop()
  print("Current Node Index, Discriminator: ", str(current.data_idx) + ", " + str(current.discriminator))

  if current.lchild != None:
    print(f"Left child: {current.lchild.data_idx}")
    node_stack.push(current.lchild)
  if current.rchild != None:
    print(f"Right child: {current.rchild.data_idx}")
    node_stack.push(current.rchild)


# test find kNN
query = [7,7]
nn = kd_tree.find_kNN(k=4, query_vec = query)

# print in order from closest to furthest (nn array is returned as max heap)
ordered_nn = sorted(nn, reverse=True)
print(ordered_nn)