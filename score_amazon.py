#!/usr/bin/env python
import networkx as nx
import os.path
import numpy
import pdb
import sys
import math
import collections
from collections import Counter
from random import choice
import random 
from scipy.optimize import minimize
import time
import numpy.linalg as LA
import multiprocessing as mp 


def f1score(nodes, heatnodes):
    common = len(nodes.intersection(heatnodes))
    precision = 1.0 * common / len(nodes)
    recall = 1.0 * common / len(heatnodes)
    if precision + recall == 0:
        return 0
    else:
        return ( 2.0 * precision * recall / (precision + recall) )


class LoadGraph:
    def __init__(self):
        self.G = nx.read_gpickle("./input/amazon/amazon.gpickle")

    def fmeasure_top5000(self):
        fg_membership = {} # the detected membership 
        fg_gnc = {} # the detected comm

        stats = []
        with open(sys.argv[1], "r+") as txt:
            for i,line in enumerate(txt):
                fg_gnc[i] = [int(_) for _ in line.split()]
                for n in fg_gnc[i]:
                    if n in fg_membership:
                        fg_membership[n].append( i )
                    else:
                        fg_membership[n] = [ i ]

        for i,nodes in self.G.graph["top5000"].items():
            print i 
            f1score_list = []
            n_comm = [ fg_membership[n] for n in nodes if n in fg_membership ]
            if n_comm:
                for comm in set(reduce(lambda x,y:x+y, n_comm)):
                    thenodes = fg_gnc[comm] 
                    f1score_list.append( f1score(set(thenodes), set(nodes)) )
                stats.append( max(f1score_list) ) 
        print 1.0 * sum(stats)/len(stats) 

if __name__ == "__main__":
    global algorithm
    print "start"
    amazon = LoadGraph()
    amazon.fmeasure_top5000() 
