#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import logging
import sys
import math
from io import open
from os import path
from time import time
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from collections import defaultdict
from multiprocessing import cpu_count
import random
from random import shuffle
from itertools import product,permutations,chain
import collections
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool
from multiprocessing import cpu_count

#novas importações
import numpy as np
import operator

class DiGraph(defaultdict):
  """Efficient basic implementation of nx `DiGraph' â€“ Directed graphs with self loops"""  
  def __init__(self):
    super(DiGraph, self).__init__(list)

  def nodesAll(self):
    successor = list(chain.from_iterable(list(self.values())))
    rootnodes = list(self.keys())
    return list(set(rootnodes + successor))

  def nodes(self):
    return list(self.keys())

  def adjacency_iter(self):
    return self.items()

  def subgraph(self, nodes={}):
    subgraph = DiGraph()

    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]

    return subgraph

  def make_undirected(self):

    t0 = time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)

    t1 = time()
    #logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))

    t1 = time()
    #logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    #self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1

    t1 = time()

    #logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True

    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def maxDegree(self):
    return max([len(self[v]) for v in list(self.keys())])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    
  
  def orderAll(self):
    return len(self.nodesAll)

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order() 

  def gToDict(self):
    d = {}
    for k,v in self.items():
      d[k] = v
    return d

  def printAdjList(self):
    for key,value in self.items():
      print (key,":",value)

def load_edgelist_signed_directed(file_, undirected=False):
  Gpi = DiGraph() # graph with plus edge in
  Gmi = DiGraph() # graph with minus edge in
  Gpo = DiGraph() # graph with plus edge out
  Gmo = DiGraph() # graph with minus edge out
  with open(file_) as f:
    for line in f:
#       import pdb; pdb.set_trace()
      if(len(line.strip().split()[:3]) > 2):
        x, y, s = line.strip().split()[:3]
        x = int(x)
        y = int(y)
        s = int(s)
      #   if s > 0:
      #     Gpi[x].append(y) # BUG!!!
      #     Gpo[y].append(x)
      #     if undirected:
      #       Gpi[y].append(x)
      #       Gpo[x].append(y)
      #   else:
      #     Gmi[x].append(y)
      #     Gmo[y].append(x)
      #     if undirected:
      #       Gmi[y].append(x)
      #       Gmo[x].append(y)
      # elif (len(line.strip().split()[:3]) == 2):
      #   x, y = line.strip().split()[:2]
      #   x = int(x)
      #   y = int(y)
      #   Gpi[x].append(y)
      #   Gpo[y].append(x)
      #   if undirected:
      #     Gpi[y].append(x)
      #     Gpo[x].append(y)
        if s > 0:
          Gpo[x].append(y)
          Gpi[y].append(x)
          if undirected:
            Gpo[y].append(x)
            Gpi[x].append(y)
        else:
          Gmo[x].append(y)
          Gmi[y].append(x)
          if undirected:
            Gmo[y].append(x)
            Gmi[x].append(y)
      elif (len(line.strip().split()[:3]) == 2):
        x, y = line.strip().split()[:2]
        x = int(x)
        y = int(y)
        Gpo[x].append(y)
        Gpi[y].append(x)
        if undirected:
          Gpo[y].append(x)
          Gpi[x].append(y)
      # else:
      #   x = line.strip().split()[:2]
      #   x = int(x[0])
      #   Gp[x] = []
      #   Gm[x] = [] # TBC

  Gpi.make_consistent()
  Gmi.make_consistent()
  Gpo.make_consistent()
  Gmo.make_consistent()
  return Gpi, Gmi, Gpo, Gmo
