# coding: utf-8
#__author__: Xiaoyan Lu

from igraph import Graph
from sklearn import metrics
import shutil

display = {"ARI":{0:[], 1:[]}, "AMI":{0:[], 1:[]}}
validate_foot_ball_only = True

class Execution:
  def results(self, algo, filename):
    title = self.__class__.__name__
    #=====Community Detection====#
    # weighted graph 
    print "Weighted: %s" % algo 
    vd = algo(self.g, weights = [ (lambda w: max(w,0.001) )(w) for w in self.g.es["weight"]] )
    try:
      vc1 = vd.as_clustering()
    except:
      vc1 = vd #in case a VertexCluster instance is returned
    self.write_vertex_clustering(vc1, "_weighted_%s" % filename)

    # unweighted graph
    print "Un-weighted: %s" % algo
    vd = algo(self.g)
    try:
      vc2 = vd.as_clustering()
    except:
      vc2 = vd #in case a VertexCluster instance is returned
    self.write_vertex_clustering(vc2, "_unweighted_%s" % filename)

    #=====Validation====#
    if "football" in filename and validate_foot_ball_only:
      self.quality(vc1, vc2)

  def write_vertex_clustering(self, vc, tag):
      title = "community" 
      with open("./output/" + title + tag + ".group", "w+") as txt:
        for cc in range(len(vc)):
            if len(vc[cc]) > 1:
          	txt.write(" ".join([str(self.reverse_id(_)) for _ in vc[cc]]) + "\n")
      print "Write to file", "./output/" + title + tag + ".group"
  
  def quality(self, vc1, vc2):
    global display

    with open("./input/football_raw_data/footballTSEinputConference.clu", "r") as fconf:
      conf = (fconf.read()).split()
    self.g.vs["comm"] = [x for x in conf]

    for index, vc in enumerate([vc1, vc2]):
      print "=================="
      for cc in range(len(vc)):
        for cci in vc[cc]:
          self.g.vs[ self.reverse_id(cci) ]["detected"] = str(cc)
      ami = metrics.adjusted_mutual_info_score(self.g.vs["detected"], self.g.vs["comm"])
      print "AMI", ami
      display["AMI"][index].append(ami)

      ari = metrics.adjusted_rand_score(self.g.vs["detected"], self.g.vs["comm"])
      print "ARI", ari
      display["ARI"][index].append(ari)

  def compact_id(self, i):
    if i in self.ID:
      return self.ID[i]
    else:
      self.ID[i] = self.cap
      self.rID[self.cap] = i
      self.cap += 1
      return self.ID[i]

  def reverse_id(self, i):
    assert i in self.rID
    return self.rID[i]

  def __init__(self, filename):
    print "Load file", filename
    self.cap = 0
    self.ID = {} 
    self.rID = {} 
    edges = []
    weights = []
    with open(filename, "r") as txt:
      for line in txt:
        seg = line.split()
        edges.append( (self.compact_id(int(seg[0])), self.compact_id(int(seg[1]))) )
        if len(seg) == 3:
          weights.append( float(seg[2]) )
	else:
          weights.append(1) 
    self.g = Graph()
    self.g.add_vertices(self.cap)
    self.g.add_edges(edges)
    self.g.to_undirected()
    self.g.es["weight"] = weights


if __name__ == "__main__":
  print "Start"
  #Execution("./output/amazon/com-amazon_weight.pairs").results(Graph.community_fastgreedy, "amazon")
  Execution("./output/amazon.wpairs").results(Graph.community_fastgreedy, "amazon")

  exit(1)

  #===========LFR Network==============
  #mu = "mu0.45"
  #for i in range(10):
  #  print mu, i
  #  Execution("./LFR/%s/instance%d/LFR.wpairs" %( mu, i) ).results(Graph.community_fastgreedy, "LFR")
  #  # for 10 LFR instances  
  #  shutil.move("./output/community_unweighted_LFR.group", "./LFR/%s/instance%d/LFR_unweighted.group" %( mu, i ) )
  #  shutil.move("./output/community_weighted_LFR.group", "./LFR/%s/instance%d/LFR_weighted.group" %( mu, i ) )

  #exit(1)

  #=========Football Network===========
  algos = [Graph.community_fastgreedy, \
      Graph.community_label_propagation, \
      Graph.community_leading_eigenvector, \
      Graph.community_walktrap, \
      Graph.community_edge_betweenness, \
      Graph.community_multilevel]

  for index, alg in enumerate(algos):
    print alg
    Execution("./output/football.wpairs").results(alg, "football%d" % index)

  print "=====Display for Latex====="
  print "AMI"
  print "&".join(["%.5f" % _ for _ in display["AMI"][1]])
  print "&".join(["%.5f" % _ for _ in display["AMI"][0]])
  print "ARI"                                           
  print "&".join(["%.5f" % _ for _ in display["ARI"][1]])
  print "&".join(["%.5f" % _ for _ in display["ARI"][0]])
