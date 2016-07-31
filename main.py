#!/bin/python
# Author:  <>
# Description: the program for supervised modularity maximaziation
'''
@author: Xiaoyan Lu
@organization: RPI
@copyright: 2016-2017 Xiaoyan Lu
@license: MIT
@contact: lux5@rpi.com
@note: This is the program for supervised modularity maximaziation. The idea is to assign weights to edges according to their their topogical features so that the modularity is maximaziation when desired community gets detected.
@requires: U{sklearn <http://scikit-learn.org/stable/>}, U{iGraph <http://igraph.org/python/>}
@attention: Please cite our paper if you use this code.
'''

import numpy as np
import random
import matplotlib.pyplot as plt
from igraph import Graph
import sys
from scipy.optimize import minimize
from operator import itemgetter
import time
from numpy import linalg as LA
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import linecache
from sklearn import metrics
import math
import operator 

###################Configurations######################
lossfunction_const = 0
regularization_const = (0.1, 1)
inputdata = "Football"
solver = "BFGS"
totalfeatures = 2

algorithm = None # the algorithm created as global variable

###########################################################
######################### UTIL ############################
###########################################################

#################### LOSS FUNCTIONS #######################
def h(x):
  '''
  the loss function
  '''
  return max(x - lossfunction_const, 0) ** 2.0

def h_der(x):
  '''
  the first order derivative of the loss function
  @param x: input variable
  '''
  if x <= lossfunction_const:
    return 0
  else:
    return 2 * (x - lossfunction_const)
  
def link(x):
  '''
  @return : the link function
  '''
  global linkfunction
  if linkfunction=="inverse":
    return 1.0/x
  elif linkfunction=="log":
    return math.exp(x)
  elif linkfunction=="logit":
    return math.exp(x) / (1.0 + math.exp(x))

def link_der(x):
  '''
  @return : the first order derivative of the link function
  @param x: the input variable
  '''
  global linkfunction
  if linkfunction=="inverse":
    return - 1.0 / (x**2.0)
  elif linkfunction=="log":
    return math.exp(x)
  elif linkfunction=="logit":
    return math.exp(x) / ( (1.0 + math.exp(x)) ** 2.0 )

def f_wrapper(p):
  global algorithm
  return algorithm.f(p)

def f_der_wrapper(p):
  global algorithm
  return algorithm.f_der(p)

def solve():
  global algorithm
  starttime = time.time()
  res = minimize(f_wrapper, algorithm.p, method=solver, jac=f_der_wrapper, \
                options={'disp': True})
  endtime = time.time()
  print "time of convergence:",
  print(endtime - starttime)
  print "p=", (res.x)

############################################################
########################ALGORITHM###########################
############################################################
############################################################
class AdaptiveMM:
  """Adaptive Modularity Maximaziation Class"""

  ########################global variables and configurations################################################
  X = None   #:   the feature matrix, each column for one edge's features
  XColSum = None #: a column vector which is equal to the summation of all columns of X
  C = None #: a list of pairs of labeled communities, i.e. the training set
  lambda1 = None #: paramater used by the penalty method
  number_features = totalfeatures #6 #: number of features of each edge
  crucial_weights = [] #: a list of the edges with at least one end in the labeled communities
  g = None
  p = None
  linkfunction = np.vectorize(link)
  linkfunction_der = np.vectorize(link_der)

  #########################EDGE FEATURE FUNCTIONS####################################################################
  def sum_weight(self):
    '''
    @return: the summation of weights of all edges
    '''
    return (self.XColSum.T).dot(self.p) #@TODO
    #return sum( self.g.es["weight"] )

  def edge_features(self):
    '''
    @param g: the igraph Graph instance
    @return: the features matrix
    '''
    edges = self.g.get_edgelist()
    self.number_features = totalfeatures
    self.X = np.zeros((self.number_features, len(edges)))
    for index, e in enumerate(edges):
      u = e[0]
      v = e[1]
      if self.number_features == 6:
        attr_list = [ \
        max(self.g.strength([u])[0] ,self.g.strength([v])[0]) \
       ,self.g.similarity_jaccard(pairs=[(u,v)])[0] \
       ,self.g.similarity_dice(pairs=[(u,v)])[0] \
       ,self.g.transitivity_local_undirected(vertices=[u,v])[0] \
       ,self.g.transitivity_local_undirected(vertices=[u,v])[1] \
         ,1 \
        ]
      elif self.number_features == 2:
        attr_list = [ \
        1.0 * len( list( set( self.g.neighbors(u) ).intersection( set( self.g.neighbors(v) ) ) ) ) \
         ,1 \
        ]
      elif self.number_features == 3:
        attr_list = [ \
        1.0 * len( list( set( self.g.neighbors(u) ).intersection( set( self.g.neighbors(v) ) ) ) ) \
        ,self.g.similarity_jaccard(pairs=[(u,v)])[0] \
         ,1 \
        ]
      self.X[:,index] = np.array(attr_list)
    self.XColSum = self.X.sum(axis=1)

    '''
      #Full list of features extraction functions in iGraph
      attr_list = [ \
      #len(set( g.neighbors(u) ).intersection( set( g.neighbors(v) ) )) \
      #1.0 * len( list( set( self.g.neighbors(u) ).intersection( set( self.g.neighbors(v) ) ) ) ) / len(self.g.neighbors(u)) \
      #,1.0 * len( list( set( self.g.neighbors(u) ).intersection( set( self.g.neighbors(v) ) ) ) ) / len(self.g.neighbors(v)) \
      #,len(set( g.neighbors(u) ).intersection( set().union(*[ g.neighbors(ng) for ng in g.neighbors(v)] ) - set(g.neighbors(v)) - set([v]) )) \
      max(self.g.strength([u])[0] ,self.g.strength([v])[0]) \
       #,1.0 * g.strength([u])[0] / g.maxdegree() \
       #,1.0 * g.strength([v])[0] / g.maxdegree() \
       #,g.strength([u])[0] \
       #,g.knn(vids=[u])[0][0] \
     ,self.g.similarity_jaccard(pairs=[(u,v)])[0] \
     ,self.g.similarity_dice(pairs=[(u,v)])[0] \
  #    ,print g.similarity_inverse_log_weighted(vertices=[u,v])
  # @TODO edge_disjoint_paths
     ,self.g.transitivity_local_undirected(vertices=[u,v])[0] \
     ,self.g.transitivity_local_undirected(vertices=[u,v])[1] \
       ,1 \
      ]
    '''

  def weight_linear_regression(self):
    '''
    @return: weight computed by linear regression model
    '''
    self.g.es["weight"] = ((self.X).T).dot(self.p) 
    #self.g.es["weight"] = self.linkfunction(  ((self.X).T).dot(self.p)  ) 

  ################## MODULARITY CHANGE FUNCTIONS #####################
  def get_crucial_weights(self):
    '''
    Compute the indices of edges which are inside or between labeled community pairs
    In this way, the functions, W_IN, W_ALL, W_BETWEEN get accelerated
    @param C: a list of pairs of labeled communities as training set for supervised learning
    '''
    self.g["CEdgeList"] = {}
    for index in range(len(self.C)):
      self.g["CEdgeList"][index] = {"out_c1":set(), "in_c1":set(), "between12":set(), "in_c2":set(), "out_c2":set()}
    for index, (c1,c2) in enumerate(self.C):
      for v in c1:
        for vnh in self.g.neighbors(v):
          if vnh in c1:
            self.g["CEdgeList"][index]["in_c1"].add(self.g.get_eid(v,vnh))
          else:
            self.g["CEdgeList"][index]["out_c1"].add(self.g.get_eid(v,vnh))
            if vnh in c2:
              self.g["CEdgeList"][index]["between12"].add(self.g.get_eid(v,vnh))
      for v in c2:
        for vnh in self.g.neighbors(v):
          if vnh in c2:
            self.g["CEdgeList"][index]["in_c2"].add(self.g.get_eid(v,vnh))
          else:
            self.g["CEdgeList"][index]["out_c2"].add(self.g.get_eid(v,vnh))
            
  def sum_weight_edges(self, eids):
    '''
    @return: the summation of weights of edges, given a list of edge ids
    '''
    ret = 0
    for eid in eids:
      ret += self.g.es[eid]["weight"]
    return ret

  def delta_q(self, index):
    '''
    @return: modularity change upon joining two communities
    @param index: the index of the community pair in C
    @param p: latent variables p for regression
    @param g: the igraph Graph instance
    '''
    sum_W = self.sum_weight()
    W_OUT_c1 = self.sum_weight_edges(self.g["CEdgeList"][index]["out_c1"])
    W_OUT_c2 = self.sum_weight_edges(self.g["CEdgeList"][index]["out_c2"])
    Double_W_IN_c1 = 2.0 * self.sum_weight_edges(self.g["CEdgeList"][index]["in_c1"])
    Double_W_IN_c2 = 2.0 * self.sum_weight_edges(self.g["CEdgeList"][index]["in_c2"])
    W_ALL_c1 = Double_W_IN_c1 + W_OUT_c1
    W_ALL_c2 = Double_W_IN_c2 + W_OUT_c2
    W_BETWEEN12 = self.sum_weight_edges(self.g["CEdgeList"][index]["between12"])
    delta = 2.0 * ( W_BETWEEN12 / 2.0 / sum_W - W_ALL_c1 * W_ALL_c2 / 4.0 / (sum_W * sum_W) ) 
    return delta

  def delta_q_over_w(self, index, edgetype):
    '''
    @return: partial derivative of modularity change over the weight of edge with index "eid"
    @param index: the index of community pairs
    @param edgetype: the type of the edge with respect to the community pairs
    '''
    sum_W = self.sum_weight()
    W_OUT_c1 = self.sum_weight_edges(self.g["CEdgeList"][index]["out_c1"])
    W_OUT_c2 = self.sum_weight_edges(self.g["CEdgeList"][index]["out_c2"])
    Double_W_IN_c1 = 2.0 * self.sum_weight_edges(self.g["CEdgeList"][index]["in_c1"])
    Double_W_IN_c2 = 2.0 * self.sum_weight_edges(self.g["CEdgeList"][index]["in_c2"])
    W_ALL_c1 = Double_W_IN_c1 + W_OUT_c1
    W_ALL_c2 = Double_W_IN_c2 + W_OUT_c2
    W_BETWEEN12 = self.sum_weight_edges(self.g["CEdgeList"][index]["between12"])

    ret = (1.0 * W_ALL_c1 * W_ALL_c2 - sum_W * W_BETWEEN12) / sum_W ** 3.0
    if edgetype == "out_c1":
      ret += - 1.0 * W_ALL_c2 / 2.0 / sum_W ** 2
    elif edgetype == "in_c1":
      ret += - 1.0 * W_ALL_c2 / sum_W ** 2
    elif edgetype == "between12":
      ret += 1.0 / sum_W
    elif edgetype == "in_c1":
      ret += - 1.0 * W_ALL_c1 / sum_W ** 2
    elif edgetype == "in_c1":
      ret += - 1.0 * W_ALL_c1 / sum_W ** 2
    return ret

  ################## OBJECTIVE FUNCTION AND ITS DERIVATIVES #####################
  def f_w(self):
    '''
    @return : the objective function defined by weights
    '''
    ret = 0
    for index in range(len(self.C)):
        #print "Delta Q", index, "=", self.delta_q(index)
        ret += self.lambda1 * h( self.delta_q(index) )
    return ret

  def f_over_w(self, index, edgetype):
    '''
    @return: first order derivative of the objective function over weight w of edge of type "edgetype"
    '''
    dDeltaQ_dw = self.delta_q_over_w(index, edgetype)
    dh_dDeltaQ = h_der( self.delta_q(index) )
    return self.lambda1 * dh_dDeltaQ * dDeltaQ_dw

  def f_over_p(self):
    '''
    @return: first order derivative of the objective function over latent variable p
    '''
    #df_w = np.zeros((len(self.g.es), 1))
    #only for relevant edges: connected to at least one node inside the clusterings    
    for index in range( len(self.C) ):
      df_w = np.full((len(self.g.es), 1), self.f_over_w(index, "others"))
      df_dw = { "out_c1":self.f_over_w(index, "out_c1"),\
                "out_c2":self.f_over_w(index, "out_c2"),\
                "in_c1":self.f_over_w(index, "in_c1"),\
                "in_c2":self.f_over_w(index, "in_c2"),\
                "between12":self.f_over_w(index, "between12")\
      }
      for edge in self.g["CEdgeList"][index]["out_c1"]:
        df_w[edge] = df_dw["out_c1"]
      for edge in self.g["CEdgeList"][index]["out_c2"]:
        df_w[edge] = df_dw["out_c2"]
      for edge in self.g["CEdgeList"][index]["in_c1"]:
        df_w[edge] = df_dw["in_c1"]
      for edge in self.g["CEdgeList"][index]["in_c2"]:
        df_w[edge] = df_dw["in_c2"]
      for edge in self.g["CEdgeList"][index]["between12"]:
        df_w[edge] = df_dw["between12"]
    df_p = (self.X).dot( df_w )
    return df_p
   
  def f(self, pval):
    '''
    the objective function
    @note: loss functin h(x)
    '''
    self.p = pval
    self.weight_linear_regression()
    ret = self.f_w() \
          + regularization_const[0] * ( (self.sum_weight() - len(self.g.es)) ** 2.0 / len(self.g.es) ) \
          + regularization_const[1] * LA.norm(self.p, 2)
    #print "f=", ret
    #print "p=", self.p
    #print "summation of weight=", self.sum_weight()
    #print "number of edges=", len(self.g.es)
    #print "-" * 20
    return ret

  def f_der(self, pval):
    '''
    @return: first order derivative of the objective function
    '''
    self.p = pval
    self.weight_linear_regression()
    ret = self.f_over_p() \
          + regularization_const[0] * (  2.0 * ( self.sum_weight() - len(self.g.es) ) / len(self.g.es) * (self.XColSum)  ) \
          + regularization_const[1] * (self.p / LA.norm(self.p, 2))

    print "|df/dp|=",  LA.norm(ret)
    return ( ret.T )[0]

  ###########################GENERATING LABELED COMMUNITY AND INIT####################################

  def __init__(self, g, Cval, p0 = None, lambda0 = None, feature = None): 
    '''
    Initialize the parameters, constants for the optimization problem
    @note: it must be called before solving the optimization
    '''
    self.g = g
    self.edge_features()
    self.C = Cval
    self.get_crucial_weights()
    if lambda0:
      self.lambda1 = lambda0
    else:
      self.lambda1 = 200000 # the default inital value for co-efficient lambda
    if p0 == None:
      self.p = np.ones((self.number_features,1)) # the default inital value for p
    else:
      self.p = p0
    self.weight_linear_regression()

###########################GRAPH LOADING AND EXECUTION####################################

class BaseInputGraph:
  def run(self, C, p0 = None):
    """
    Solve the optimization problem 
    and return the edge weights
    @param C: labeled communities used as the training set
    """
    global algorithm 
    algorithm = AdaptiveMM(self.g, C, p0 = p0, lambda0 = 2000)
    solve()

  def write_vertex_clustering(self, vc, tag):
      title = self.__class__.__name__
      with open(title + tag + ".group", "w+") as txt:
        for cc in range(len(vc)):
          txt.write(" ".join([str(_) for _ in vc[cc]]) + "\n")

  def results(self, algo, hasgnc = False, filename="_"):
    title = self.__class__.__name__
    AMI_increase = []
    ARI_increase = []
    rounds = 1
    if hasgnc: rounds = 10
    print "Runing ", algo.__name__, "for", rounds, "rounds"
    for i in range(rounds):
      vd = algo(self.g, weights = [ (lambda w: max(w,0) )(w) for w in self.g.es["weight"]] )
      try:
        vc = vd.as_clustering()
      except:
        vc = vd #in case a VertexCluster instance is returned
      self.write_vertex_clustering(vc, "_weighted%s" % filename)
      if hasgnc:
        for cc in range(len(vc)):
          for cci in vc[cc]:
            self.g.vs[cci]["fastgreedy_withweight"] = str(cc)
      vd = algo(self.g)
      try:
        vc = vd.as_clustering()
      except:
        vc = vd #in case a VertexCluster instance is returned
      self.write_vertex_clustering(vc, "_unweighted%s" % filename)
      if hasgnc:
        for cc in range(len(vc)):
          for cci in vc[cc]:
            self.g.vs[cci]["fastgreedy_withoutweight"] = str(cc)
        #self.g.write_gml("%s.gml" % title)
        #print "%s.gml written with attributes" % title,
        #print self.g.vs.attributes()
      if hasgnc:
        #print "Weighted:"
        #print "Adjusted Mutual Information:", 
        ami_weight = metrics.adjusted_mutual_info_score(self.g.vs["fastgreedy_withweight"], self.g.vs["comm"])
        #print "Adjusted Rand index:", 
        ari_weight = metrics.adjusted_rand_score(self.g.vs["fastgreedy_withweight"], self.g.vs["comm"])
        #print "~"*30
        #print "Unweighted:"
        #print "Adjusted Mutual Information:", 
        ami_unweight = metrics.adjusted_mutual_info_score(self.g.vs["fastgreedy_withoutweight"], self.g.vs["comm"])
        #print "Adjusted Rand index:", 
        ari_unweight = metrics.adjusted_rand_score(self.g.vs["fastgreedy_withoutweight"], self.g.vs["comm"])

        AMI_increase.append(ami_weight - ami_unweight)
        ARI_increase.append(ari_weight - ari_unweight)
    if hasgnc:
      print "Adjusted Mutual Information increases by",
      print 1.0 * sum(AMI_increase) / len(AMI_increase)
      print "Adjusted Rand index increases by",
      print 1.0 * sum(ARI_increase) / len(ARI_increase)
      print "-" * 20
      return AMI_increase
      #return ARI_increase

  def get_C(self, group):
    '''
    @param group: a list of nodes as ONE ground truth communities
    @return: the labeled communities as the training set for supervised learning
    '''
    fl = []
    for l in self.g.neighborhood(vertices=group):
      fl += list(set(l) - set(group))
    c = Counter(fl)
    C = []
    #select the top-3 neighbors with the most connections to the community
    for cc1, freq in c.most_common(3):
      C.append( (group, [cc1]) )
    print "The supervised set(None-merge):", C
    return C

  def write_ground_truth(self, gtfilename):
    '''
    write the ground truth communities to a text file,
    one community per line
    '''
    comms = {}
    for v, c in enumerate(self.g.vs["comm"]):
      if not c in comms:
        comms[c] = set()
      comms[c].add(v)
    txt = open(gtfilename, "w+")
    for c, s in comms.items():
      for ele in s:
        txt.write(str(ele)+" ")
      txt.write("\n")
    txt.close()

  def unsupervised_logexpand(self):
    n = len(self.g.vs)
    seedlist = [random.randint(0,n) for _ in range(int(math.sqrt(n)))]
    hood = {}
    diameterchange = {}
    for seed in seedlist:
      try:
        nns = self.g.neighbors(seed)
      except:
        continue #in case of wrong seed
      nn2 = -1
      for nn in nns:
        intersection = ( set(self.g.neighbors(nn)) - set([seed]) ) & ( set(nns) )
        if len(intersection) > 0:
          nn2 = list(intersection)[0]
          break
      if nn2 == -1: continue
      hood[seed] = [seed, nn, nn2]
      fl = []
      for l in self.g.neighborhood(vertices=hood[seed]):
        fl += list(set(l) - set(hood[seed]))
      c = Counter(fl)
      dist = dict( c )
      subg = self.g.induced_subgraph(hood[seed])
      d = subg.diameter()
      prevd = 1
      for i in range(20):
        try:
          node = max(dist.iteritems(), key=operator.itemgetter(1))[0]
          hood[seed].append( node )
          del dist[node]
          for nodeneighbor in ( set(self.g.neighbors(node)) - set(hood[seed]) ):
            if nodeneighbor in dist:
              dist[nodeneighbor] += 1
            else:
              dist[nodeneighbor] = 1
          if len(hood[seed]) == 6:
            d = self.g.induced_subgraph(hood[seed]).diameter()
            if d >= 2: continue #give up the community
          if len(hood[seed]) == 12:
            d = self.g.induced_subgraph(hood[seed]).diameter()
            if d >= 3: continue #give up the community
          d = self.g.induced_subgraph(hood[seed]).diameter()
          if prevd == 2 and d == 3:
            diameterchange[seed] = len(hood[seed]) #save the community
            break
          prevd = d
        except:
          break
    bestseeds = dict(sorted(diameterchange.iteritems(), key=operator.itemgetter(1), reverse=True)[:20])
    #bestseeds = [ random.choice(diameterchange.keys()) for _ in range(20)]
    print "bestseeds", bestseeds
    C = []
    for seed in bestseeds.keys():
    #for seed in bestseeds:
      cc1 = hood[seed][-1]
      hood[seed].remove(cc1)
      C.append( (hood[seed], [cc1] ) )
    print "C=", C
    return C

class Ring ( BaseInputGraph ):
  wvalue = 1
  g = Graph() # the iGraph graph instance

  def __init__(self, v):
    '''
    @param v: the weight of edge connecting two cliques
    @return: the produced ring network as iGraph graph instance 
    '''
    self.wvalue = v
    edges = []
    ws = []
    clique_num = 30
    clique_size = 5
    for c in range(0, clique_num):
      for i in range(clique_size*c, clique_size*(c+1)):
        for j in range(i+1, clique_size*(c+1)):
          edges.append((i,j))
          ws.append(1)
    for c in range(0, clique_num):
          edges.append((clique_size*c, clique_size*( (c+1) % clique_num)))
          ws.append(self.wvalue)
    maxid = max( edges, key=itemgetter(1))[1]
    maxid = max( maxid, max(edges,key=itemgetter(0))[0] )
    
    self.g = Graph()
    self.g.add_vertices(maxid + 1)
    self.g.add_edges(edges)
    self.g.es["weight"] = ws
    self.g.vs["comm"] = [str( int(_ / clique_size) ) for _ in range(len(self.g.vs))]
    print "#nodes=", maxid + 1
    print "#edges=", len(self.g.es)

  def run(self):
    '''
    run the algorithm
    '''
    #supervised
    group = [0,1,2,3,4]
    C = BaseInputGraph.get_C(self, group)
    #unsupervised
    #C = BaseInputGraph.unsupervised_logexpand(self)
    print C
    BaseInputGraph.run(self, C)
    for e in self.g.es:
      print "(",e.tuple[0]," ,",
      print e.tuple[1],")=",
      print e["weight"]
    #do not know why, but the fast greedy algorithm implemented in iGraph sometimes has some problems.
    #The original implementation in c++ released by Dr.Newman on his website always works well.

class Football ( BaseInputGraph ):
  def __init__(self):
    '''
    @return: American Colleage Football Network as iGraph graph instance 
    '''
    edges = []
    weights = []
    f = open("./football/footballTSEinputEL.dat", "r")
    for line in f:
      seg = line.split()
      edges.append( (int(seg[0]), int(seg[1])) )
      weights.append( 1 )
    maxid = max( edges, key=itemgetter(1))[1]
    maxid = max( maxid, max(edges,key=itemgetter(0))[0] )
    self.g = Graph()
    self.g.add_vertices(maxid + 1)
    self.g.add_edges(edges)
    self.g.to_undirected()
    conf = []
    with open("./football/footballTSEinputConference.clu", "r") as fconf:
      conf = (fconf.read()).split()
    self.g.vs["comm"] = [x for x in conf]
    self.g.vs["myID"] = [ str(int(i)) for i in range(maxid+1)]
    print "#nodes=", maxid + 1
    print "#edges=", len(self.g.es)

  def run(self):
    AMI_increase = [[], [], [], [], [], []]
    for igroup in range(10):
      #semi-supervised
      #group = [i for i, conf in enumerate(self.g.vs["comm"]) if conf == str(igroup)]
      #if len(group) < 3: continue
      #C = BaseInputGraph.get_C(self, group)

      #unsupervised
      C = BaseInputGraph.unsupervised_logexpand(self)

      BaseInputGraph.run(self, C)
      AMI_increase[0] += BaseInputGraph.results(self, Graph.community_fastgreedy, hasgnc = True)
      AMI_increase[1] += BaseInputGraph.results(self, Graph.community_label_propagation, hasgnc = True)
      AMI_increase[2] += BaseInputGraph.results(self, Graph.community_leading_eigenvector, hasgnc = True)
      AMI_increase[3] += BaseInputGraph.results(self, Graph.community_walktrap, hasgnc = True)
      AMI_increase[4] += BaseInputGraph.results(self, Graph.community_edge_betweenness, hasgnc = True)
      AMI_increase[5] += BaseInputGraph.results(self, Graph.community_multilevel, hasgnc = True)
    for i in range(6):
      print "& %.5f" % ( 1.0 * sum(AMI_increase[i]) / len(AMI_increase[i]) )

class LFR ( BaseInputGraph ):
  def __init__(self, trialval=1):
    ws = []
    edges = []
    self.trial = trialval
    with open("./binary_networks/mu0.5/network%d.dat" % self.trial, "r") as txt:
      for line in txt:
        seg = line.split()
        edges.append((int(seg[0]), int(seg[1])))
        ws.append(1)
    maxid = max( edges, key=itemgetter(1))[1]
    maxid = max( maxid, max(edges,key=itemgetter(0))[0] )
    self.g = Graph()
    print maxid
    self.g.add_vertices(maxid + 1)
    with open("./binary_networks/mu0.5/community%d.dat" % self.trial, "r") as txt:
      for line in txt:
        seg = line.split()
        #print seg[0]
        self.g.vs[int(seg[0])]["comm"] = seg[1] #note: string is returned
    self.g.add_edges(edges)
    self.g.to_undirected()
    self.g.simplify()
    self.g.delete_vertices(0)
    self.g.es["weight"] = ws
    BaseInputGraph.write_ground_truth(self, "./ground_truth_community%d.groups" % self.trial)
    print "#nodes=", maxid + 1
    print "#edges=", len(self.g.es) 

  def run(self):
    #supervised
    C = []
    for i in range(6):
      commval = str(random.randint(0,100))
      group = [i for i, comm in enumerate(self.g.vs["comm"]) if comm == commval]
      C += BaseInputGraph.get_C(self, group)
    #unsupervised
    C = BaseInputGraph.unsupervised_logexpand(self)
    BaseInputGraph.run(self, C, p0=np.array([1 , 1]))
    BaseInputGraph.results(self, Graph.community_fastgreedy, hasgnc = False,\
     filename="%d" %self.trial)

class Physic (BaseInputGraph):
  def __init__(self):
    '''
    @return: Arxiv ASTRO-PH (Astro Physics) collaboration network as iGraph graph instance 
    '''
    edges = []
    weights = []
    f = open("./physic/compact-physic.txt", "r")
    for line in f:
      if line and line[0]!='#':
        seg = line.split()
        edges.append( (int(seg[0]), int(seg[1])) )
        weights.append( 1 )
    maxid = max( edges, key=itemgetter(1) )[1]
    maxid = max( maxid, max(edges,key=itemgetter(0))[0] )
    self.g = Graph()
    self.g.add_vertices(maxid + 1)
    self.g.add_edges(edges)
    self.g.to_undirected()
    self.g.simplify()
    self.g.vs["myID"] = [ str(int(i)) for i in range(maxid+1)]
    print "#nodes=", maxid + 1
    print "#edges=", len(self.g.es)

  def run(self):
    C = BaseInputGraph.unsupervised_logexpand(self)
    BaseInputGraph.run(self, C, p0=np.array([0.04, 0.04]))
    with open("./physic/Physic_weights.pairs", "w+") as txt:
      for e in self.g.es:
        txt.write("%d %d %f\n" %(e.tuple[0], e.tuple[1], e["weight"]) )
    #BaseInputGraph.results(self, Graph.community_fastgreedy)

class Enron (BaseInputGraph):
  def __init__(self):
    '''
    @return: Enron email communication network as iGraph graph instance 
    '''
    edges = []
    weights = []
    f = open("./enron/email-Enron.txt", "r")
    for line in f:
      if line and line[0]!='#':
        seg = line.split()
        edges.append( (int(seg[0]), int(seg[1])) )
        weights.append( 1 )
    maxid = max( edges, key=itemgetter(1) )[1]
    maxid = max( maxid, max(edges,key=itemgetter(0))[0] )
    self.g = Graph()
    self.g.add_vertices(maxid + 1)
    self.g.add_edges(edges)
    self.g.to_undirected()
    self.g.simplify()
    self.g.vs["myID"] = [ str(int(i)) for i in range(maxid+1)]
    print "#nodes=", maxid + 1
    print "#edges=", len(self.g.es)

  def run(self):
    C = BaseInputGraph.unsupervised_logexpand(self)
    BaseInputGraph.run(self, C, p0=np.array([0.04, 0.04]))
    with open("./enron/email-Enron_weights.pairs", "w+") as txt:
      for e in self.g.es:
        txt.write("%d %d %f\n" %(e.tuple[0], e.tuple[1], e["weight"]) )
    with open("./enron/email-Enron_unweights.pairs", "w+") as txt:
      count = 0
      for e in self.g.es:
        txt.write("%d %d\n" %(e.tuple[0], e.tuple[1]) )
        count += 1
      print count , "edges written."
    #BaseInputGraph.results(self, Graph.community_fastgreedy)

if __name__ == "__main__":
  if inputdata == "Ring":
    Ring(1).run()
  elif inputdata == "Football":
    Football().run()
  elif inputdata == "LFR":
    for i in range(1,11):
      LFR(i).run()
  elif inputdata == "Amazon":
    Amazon().run()
  elif inputdata == "Physic":
    Physic().run()
  elif inputdata == "Enron":
    Enron().run()
