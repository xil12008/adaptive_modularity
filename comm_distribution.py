#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from operator import itemgetter

import sys


def data(filename):
    with open(filename, "r+") as txt:
        x = [len(line.split()) for line in txt]
    c = Counter(x)
    sc = sorted(c.items(), key=itemgetter(0), reverse=True)
    return [_[0] for _ in sc], [_[1] for _ in sc]

if __name__ == "__main__":
    #x1, y1 = data("MyGraph_unweightedamazon.group")
    #x2, y2 = data("MyGraph_weightedamazon.group")

    #x1, y1 = data("SLPAw_com-amazon.ungraph_run1_r0.01_v3_T100.icpm")
    #x2, y2 = data("SLPAw_com-amazon_weight_run1_r0.01_v3_T100.icpm")

    #x3, y3 = data("com-amazon.all.dedup.cmty.txt")
    x1,y1 = data(sys.argv[1])
    x2,y2 = data(sys.argv[2])
    x3,y3 = data(sys.argv[3])

    #comm size vs comm num
    #print x1,y1
    #print x2,y2
    #print x3,y3

    plt.plot(x1, y1, '-bx', label="Modularity (Original)")
    plt.plot(x2, y2, '-r*', label="Adaptive Modularity")
    plt.plot(x3, y3, '-g', label="Ground Truth")

    plt.legend(loc='upper right')
    plt.ylim([1, 500])
    plt.xlim([8, 150])
    #plt.xscale('log')

    plt.title('Distribution of Community Size')
    plt.xlabel('Community Size')
    plt.ylabel('Number of Communities')
    plt.savefig("comm_distribution_unweighted.pdf")
