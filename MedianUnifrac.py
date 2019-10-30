
import numpy as np
import dendropy
import re
import warnings

def aggregate(Tint,Lint,D):
    '''
    :param Tint: Indices of nodes
    :param Lint: Lengths of edges
    :param D: a vector to be transformed
    :return: transformed D
    '''
    root = max(Tint.values())
    v = root
    while v>0:
        for w in Tint.keys():
            if Tint[w] == v:
                D[v] = D[v] + D[w]
        v = v-1
    for i in range(root):
        D[i] = Lint[(i, Tint[i])] * D[i]
    return (D)

def median_of_vectors(L):
    '''
    :param L: a list of vectors
    :return: a vector with each entry i being the median of vectors of L at position i
    '''
    return np.median(L, axis=0)

def inverse_aggregate(Tint, Lint, P):
    '''
    :param Tint: indices of nodes
    :param Lint: lengths of edges
    :param P: a vector to be 'inversed'
    :return: inversed P
    '''
    root = max(Tint.values())
    v = 0
    for i in range(max(Tint.values())):
        P[i] = P[i]/Lint[(i, Tint[i])]
    while v<=root:
        for w in Tint.keys():
            if Tint[w] == v: #w is a daughter of v
                P[v] = P[v] - P[w]
        v = v+1
    return (P)

def create_env(sample_file):
    '''
    :param sample_file: a file containding ids and samples
    :return: an env_dict in the form of { id: {sample:count} }
    '''
    env_dict = dict()
    with open(sample_file) as fp:
        line = fp.readline()
        while line:
            list = line.split()
            key = list.pop(0) #get key
            env_dict[key] = dict()
            for str in list:
                m = re.match(r"(\d+).(\w+)_(\d+)", str)
                sample = m.group(2)
                if sample in env_dict[key]:
                    env_dict[key][sample] += 1
                else:
                    env_dict[key][sample] = 1
            line = fp.readline()
    fp.close()
    return env_dict