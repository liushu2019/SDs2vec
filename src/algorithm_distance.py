# -*- coding: utf-8 -*-
from ast import While
from time import time
from collections import deque
import numpy as np
import math,logging
from fastdtw import fastdtw
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from utils import *
import os
import quaternion
from sklearn.cluster import KMeans
import collections
import warnings
import random
# import cmath

def getDegreeListsVertices_complex_directed(gpi,gni,gpo,gno,vertices,calcUntilLayer):
    degreeList = {}

    for v in vertices:
        degreeList[v] = getDegreeLists_complex_directed(gpi,gni,gpo,gno,v,calcUntilLayer)

    return degreeList

def getCompactDegreeListsVertices_complex_directed(Gpi,Gni,Gpo,Gno,vertices,calcUntilLayer,opt4):
    degreeList = {}

    logging.info('TEST getCompactDegreeListsVertices_complex in') # DEBUG
    if opt4:
        for v in vertices:
            degreeList[v] = getDegreeLists_complex_directed_kmeans(Gpi,Gni,Gpo,Gno,v,calcUntilLayer)
    else:
        for v in vertices:
            degreeList[v] = getCompactDegreeLists_complex_directed(Gpi,Gni,Gpo,Gno,v,calcUntilLayer)

    return degreeList

def getCompactDegreeLists_complex_directed(Gpi,Gni,Gpo,Gno, root, calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(list(set(list(Gpi.keys()) + list(Gni.keys()) + list(Gpo.keys()) + list(Gno.keys())))) + 1)
    # maxNodeId = max(list(set(list(Gp.keys()) + list(Gn.keys()))))
    # vetor_marcacao = np.zeros((maxNodeId+1,maxNodeId+1))

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    l = {}

    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    logging.info('BFS vertex {}. in !'.format(root))
    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        d = np.quaternion(len(Gpi[vertex]), len(Gni[vertex]), len(Gpo[vertex]), len(Gno[vertex]))
        if(d not in l):
            l[d] = 0
        l[d] += 1

        for v in Gpi[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1   
        for v in Gni[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1  
        for v in Gpo[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1   
        for v in Gno[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1  

        if(timeToDepthIncrease == 0):
            list_d = []
            for degree,freq in l.items():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0].real+x[0].imag.sum())
            # listas[depth] = np.array(list_d)#,dtype=np.int32)
            listas[depth] = list_d

            l = {}

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))

    return listas


def getDegreeLists_complex_directed_kmeans(gpi,gni,gpo,gno, root, calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(max(gpi), max(gni), max(gpo), max(gno)) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1


    l = {}

    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        d = np.quaternion(len(gpi[vertex]), len(gni[vertex]), len(gpo[vertex]), len(gno[vertex]))
        if(d not in l):
            l[d] = 0
        l[d] += 1

        for v in gpi[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1   
        for v in gni[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1  
        for v in gpo[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1   
        for v in gno[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1  

        if(timeToDepthIncrease == 0):
            list_d = []
            for degree,freq in l.items():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0].real+x[0].imag.sum())
            # listas[depth] = np.array(list_d)#,dtype=np.int32)
            listas[depth] = list_d

            l = {}

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0
            if(2 == depth):
                break
    if (calcUntilLayer is None) or (calcUntilLayer > 1):
        l=deque()
        while queue:
            vertex = queue.popleft()
            timeToDepthIncrease -= 1

            # l.append(len(g[vertex]))
            l.append([len(gpi[vertex]), len(gni[vertex]), len(gpo[vertex]), len(gno[vertex])])

            for v in gpi[vertex]:
                if(vetor_marcacao[v] == 0):
                    vetor_marcacao[v] = 1
                    queue.append(v)
                    pendingDepthIncrease += 1    
            for v in gni[vertex]:
                if(vetor_marcacao[v] == 0):
                    vetor_marcacao[v] = 1
                    queue.append(v)
                    pendingDepthIncrease += 1   
            for v in gpo[vertex]:
                if(vetor_marcacao[v] == 0):
                    vetor_marcacao[v] = 1
                    queue.append(v)
                    pendingDepthIncrease += 1    
            for v in gno[vertex]:
                if(vetor_marcacao[v] == 0):
                    vetor_marcacao[v] = 1
                    queue.append(v)
                    pendingDepthIncrease += 1   

            if(timeToDepthIncrease == 0 and depth > 1):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kmeans = KMeans(n_clusters=min(16, len(l)), random_state=0).fit(np.array(l))
                    bb = kmeans.labels_
                    aa = kmeans.cluster_centers_
                    cc = collections.Counter(bb)
                    cc = dict(sorted(cc.items(), key=lambda x:x[1], reverse=True))
                    lp = []
                    for ix,x in cc.items():
                        lp.append((np.quaternion(aa[ix][0], aa[ix][1], aa[ix][2], aa[ix][3]), x))
                    listas[depth] = lp
                    l = deque()

                if (calcUntilLayer is not None) and (calcUntilLayer == depth):
                    break
                # print (pendingDepthIncrease)
                depth += 1
                timeToDepthIncrease = pendingDepthIncrease
                pendingDepthIncrease = 0
    
    t1 = time()
    logging.info('BFS vertex kmeans {}. Time: {}s'.format(root,(t1-t0)))

    return listas

def getCompactDegreeLists(g, root, maxDegree,calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(g) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1
    l = {}

    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        d = len(g[vertex])
        if(d not in l):
            l[d] = 0
        l[d] += 1

        for v in g[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    

        if(timeToDepthIncrease == 0):

            list_d = []
            for degree,freq in l.items():
                list_d.append((degree,freq))
            list_d.sort(key=lambda x: x[0])
            listas[depth] = np.array(list_d,dtype=np.int32)

            l = {}

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))

    return listas

def getDegreeLists_complex_directed(gpi,gni,gpo,gno, root, calcUntilLayer):
    t0 = time()

    listas = {}
    vetor_marcacao = [0] * (max(max(gpi), max(gni), max(gpo), max(gno)) + 1)

    # Marcar s e inserir s na fila Q
    queue = deque()
    queue.append(root)
    vetor_marcacao[root] = 1


    l = deque()

    ## Variáveis de controle de distância
    depth = 0
    pendingDepthIncrease = 0
    timeToDepthIncrease = 1

    while queue:
        vertex = queue.popleft()
        timeToDepthIncrease -= 1

        # l.append(len(g[vertex]))
        l.append(np.quaternion(len(gpi[vertex]), len(gni[vertex]), len(gpo[vertex]), len(gno[vertex])))

        for v in gpi[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    
        for v in gni[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1   
        for v in gpo[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1    
        for v in gno[vertex]:
            if(vetor_marcacao[v] == 0):
                vetor_marcacao[v] = 1
                queue.append(v)
                pendingDepthIncrease += 1   

        if(timeToDepthIncrease == 0):

            # lp = np.array(l,dtype='float')
            # lp = np.sort(lp)
            lp = np.array(sorted(l, key= lambda x: (x.real + x.imag.sum(), x.real, x.imag[0], x.imag[1], x.imag[2])))
            listas[depth] = lp
            l = deque()

            if(calcUntilLayer == depth):
                break

            depth += 1
            timeToDepthIncrease = pendingDepthIncrease
            pendingDepthIncrease = 0


    t1 = time()
    logging.info('BFS vertex {}. Time: {}s'.format(root,(t1-t0)))


    return listas

def cost_complex_vector_Euclidean_logscale_directed(a,b):
    a_ = np.log(np.array(a)+1)
    b_ = np.log(np.array(b)+1)    
    return np.linalg.norm(a_-b_)

def cost_complex_vector_Euclidean_logscale_max_directed(a,b):
    a_ = np.log(np.array(a[0])+1)
    b_ = np.log(np.array(b[0])+1)    
    # return np.linalg.norm(a_-b_)*max(a[1],b[1])
    return np.linalg.norm(a_-b_)*max(a[1][0],b[1][0])

def cost_complex_vector_Euclidean_directed(a,b):
    # print (a,b, np.linalg.norm(a-b))
    # a_ = np.log(np.array(a)+1)
    # b_ = np.log(np.array(b)+1)    
    return np.linalg.norm(a-b)

def cost_complex_vector_Euclidean_max_directed(a,b):
    # a_ = np.log(np.array(a[0])+1)
    # b_ = np.log(np.array(b[0])+1)    
    # return np.linalg.norm(a_-b_)*max(a[1],b[1])
    return np.linalg.norm(a[0]-b[0])*max(a[1][0],b[1][0])

def verifyDegrees_4axis_directed(degrees,degree_v_root, degree_list_p, degree_list_n,degree_left_bottom, degree_right_top ):#,degree_a_p,degree_b_p,degree_a_n,degree_b_n):
    # print (degree_v_root, degree_left_bottom, degree_right_top)
    pass_flag_p = False
    pass_flag_n = False
    if (degree_list_p.index(degree_left_bottom.real) == 0) and (degree_list_p.index(degree_right_top.real)+1 == len(degree_list_p)) and (degree_list_n.index(degree_left_bottom.imag) == 0) and (degree_list_n.index(degree_right_top.imag)+1 == len(degree_list_n)):
        raise StopIteration
    if (degree_list_p.index(degree_left_bottom.real) == 0) and (degree_list_p.index(degree_right_top.real)+1 == len(degree_list_p)):
        pass_flag_p = True
        pass
    elif (degree_list_p.index(degree_left_bottom.real) == 0):
        degree_now_p = degree_list_p[degree_list_p.index(degree_right_top.real)+1]
    elif (degree_list_p.index(degree_right_top.real)+1 == len(degree_list_p)):
        degree_now_p = degree_list_p[degree_list_p.index(degree_left_bottom.real)-1]
    elif (abs(degree_list_p[degree_list_p.index(degree_right_top.real)+1] - degree_v_root.real) < abs(degree_list_p[degree_list_p.index(degree_left_bottom.real)-1] - degree_v_root.real)):
        degree_now_p = degree_list_p[degree_list_p.index(degree_right_top.real)+1]
    else:
        degree_now_p = degree_list_p[degree_list_p.index(degree_left_bottom.real)-1]
    
    if (degree_list_n.index(degree_left_bottom.imag) == 0) and (degree_list_n.index(degree_right_top.imag)+1 == len(degree_list_n)):
        pass_flag_n = True
        pass
    elif (degree_list_n.index(degree_left_bottom.imag) == 0):
        degree_now_n = degree_list_n[degree_list_n.index(degree_right_top.imag)+1]
    elif (degree_list_n.index(degree_right_top.imag)+1 == len(degree_list_n)):
        degree_now_n = degree_list_n[degree_list_n.index(degree_left_bottom.imag)-1]
    elif (abs(degree_list_n[degree_list_n.index(degree_right_top.imag)+1] - degree_v_root.imag) < abs(degree_list_n[degree_list_n.index(degree_left_bottom.imag)-1] - degree_v_root.imag)):
        degree_now_n = degree_list_n[degree_list_n.index(degree_right_top.imag)+1]
    else:
        degree_now_n = degree_list_n[degree_list_n.index(degree_left_bottom.imag)-1]
    # print (degree_now_n, degree_now_p)
    # if(degree_b_p.real == -1):
    #     degree_now_p = degree_a_p.real
    # elif(degree_a_p.real == -1):
    #     degree_now_p = degree_b_p.real
    # elif(abs(degree_b_p.real - degree_v_root.real) < abs(degree_a_p.real - degree_v_root.real)):
    #     degree_now_p = degree_b_p.real
    # else:
    #     degree_now_p = degree_a_p.real
    

    # if(degree_b_n.imag == -1):
    #     degree_now_n = degree_a_n.imag
    # elif(degree_a_n.imag == -1):
    #     degree_now_n = degree_b_n.imag
    # elif(abs(degree_b_n.imag - degree_v_root.imag) < abs(degree_a_n.imag - degree_v_root.imag)):
    #     degree_now_n = degree_b_n.imag
    # else:
    #     degree_now_n = degree_a_n.imag
    
    if ((not pass_flag_n) and (not pass_flag_p) and (abs(degree_now_n - degree_v_root.imag) < abs(degree_now_p - degree_v_root.real))) or (pass_flag_p):
        # print ('N')
        assert( (degree_now_n >= degree_right_top.imag) or (degree_now_n <= degree_left_bottom.imag) ), 'Search ERROR in verifyDegrees_matrix'
        degree_now = [complex(x, degree_now_n) for x in degree_list_p[degree_list_p.index(degree_left_bottom.real): degree_list_p.index(degree_right_top.real)+1 ]]
        if (degree_now_n > degree_right_top.imag):
            degree_right_top = complex(degree_right_top.real, degree_now_n)
        else:
            degree_left_bottom = complex(degree_left_bottom.real, degree_now_n)
    else:
        # print ('P',degree_right_top)
        assert( (degree_now_p >= degree_right_top.real) or (degree_now_p <= degree_left_bottom.real) ), 'Search ERROR in verifyDegrees_matrix'
        degree_now = [complex(degree_now_p, x) for x in degree_list_n[degree_list_n.index(degree_left_bottom.imag): degree_list_n.index(degree_right_top.imag)+1 ]]
        if (degree_now_p > degree_right_top.real):
            degree_right_top = complex(degree_now_p, degree_right_top.imag)
            # print ('P',degree_right_top, )
        else:
            degree_left_bottom = complex(degree_now_p, degree_left_bottom.imag)
    degree_now.sort(key=lambda x:abs(x - degree_v_root))

    # print (degree_now, degree_left_bottom, degree_right_top)
    return degree_now, degree_left_bottom, degree_right_top

def get_vertices_4axis_directed(v,degree_v,degrees,a_vertices,degrees_sorted_ni,degrees_sorted_pi,degrees_sorted_no,degrees_sorted_po,degrees_ni,degrees_pi,degrees_no,degrees_po): # 
    '''
    degree_v: v's degree+i, v's degree-i, v's degree+o, v's degree-o quaternion style
    degrees: 4D matrix style
    a_vertices: # nodes
    '''
    a_vertices_selected = 2 * math.log(a_vertices,2)
    #logging.info("Selecionando {} próximos ao vértice {} ...".format(int(a_vertices_selected),v))
    vertices = deque()

    try:
        c_v = 0  

        for v2 in degrees[degree_v]:
            if(v != v2):
                vertices.append(v2)
                c_v += 1
                if(c_v > a_vertices_selected):
                    raise StopIteration
                a_vertices_selected -= c_v
                a_vertices_selected = a_vertices_selected//4 + 1
        minDegree = degree_v[0]
        maxDegree = degree_v[0]
        centerDegree = degree_v[0]
        c_v = 0  
        try:
            while True:
                if (degrees_sorted_pi.index(minDegree) == 0) and (degrees_sorted_pi.index(maxDegree) < len(degrees_sorted_pi) - 1) :
                    maxDegree = degrees_sorted_pi[degrees_sorted_pi.index(maxDegree)+1]
                    for v2 in degrees_pi[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_pi.index(minDegree) > 0) and (degrees_sorted_pi.index(maxDegree) == len(degrees_sorted_pi) - 1) :
                    minDegree = degrees_sorted_pi[degrees_sorted_pi.index(minDegree)-1]
                    for v2 in degrees_pi[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_pi.index(minDegree) == 0) and (degrees_sorted_pi.index(maxDegree) == len(degrees_sorted_pi) - 1) :
                    raise StopIteration
                elif abs(degrees_sorted_pi[degrees_sorted_pi.index(minDegree)-1] - centerDegree) < abs(degrees_sorted_pi[degrees_sorted_pi.index(maxDegree)+1] - centerDegree):
                    minDegree = degrees_sorted_pi[degrees_sorted_pi.index(minDegree)-1]
                    for v2 in degrees_pi[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                else:
                    maxDegree = degrees_sorted_pi[degrees_sorted_pi.index(maxDegree)+1]
                    for v2 in degrees_pi[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
        except StopIteration:
            minDegree = degree_v[1]
            maxDegree = degree_v[1]
            centerDegree = degree_v[1]
            c_v = 0  
        try:
            while True:
                if (degrees_sorted_ni.index(minDegree) == 0) and (degrees_sorted_ni.index(maxDegree) < len(degrees_sorted_ni) - 1) :
                    maxDegree = degrees_sorted_ni[degrees_sorted_ni.index(maxDegree)+1]
                    for v2 in degrees_ni[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_ni.index(minDegree) > 0) and (degrees_sorted_ni.index(maxDegree) == len(degrees_sorted_ni) - 1) :
                    # print (minDegree, maxDegree, degrees_ni, degrees_sorted_ni)
                    minDegree = degrees_sorted_ni[degrees_sorted_ni.index(minDegree)-1]
                    for v2 in degrees_ni[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_ni.index(minDegree) == 0) and (degrees_sorted_ni.index(maxDegree) == len(degrees_sorted_ni) - 1) :
                    raise StopIteration
                elif abs(degrees_sorted_ni[degrees_sorted_ni.index(minDegree)-1] - centerDegree) < abs(degrees_sorted_ni[degrees_sorted_ni.index(maxDegree)+1] - centerDegree):
                    minDegree = degrees_sorted_ni[degrees_sorted_ni.index(minDegree)-1]
                    for v2 in degrees_ni[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                else:
                    maxDegree = degrees_sorted_ni[degrees_sorted_ni.index(maxDegree)+1]
                    for v2 in degrees_ni[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
        except StopIteration:
            minDegree = degree_v[2]
            maxDegree = degree_v[2]
            centerDegree = degree_v[2]
            c_v = 0  
        try:
            while True:
                if (degrees_sorted_po.index(minDegree) == 0) and (degrees_sorted_po.index(maxDegree) < len(degrees_sorted_po) - 1) :
                    maxDegree = degrees_sorted_po[degrees_sorted_po.index(maxDegree)+1]
                    for v2 in degrees_po[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_po.index(minDegree) > 0) and (degrees_sorted_po.index(maxDegree) == len(degrees_sorted_po) - 1) :
                    minDegree = degrees_sorted_po[degrees_sorted_po.index(minDegree)-1]
                    for v2 in degrees_po[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_po.index(minDegree) == 0) and (degrees_sorted_po.index(maxDegree) == len(degrees_sorted_po) - 1) :
                    raise StopIteration
                elif abs(degrees_sorted_po[degrees_sorted_po.index(minDegree)-1] - centerDegree) < abs(degrees_sorted_po[degrees_sorted_po.index(maxDegree)+1] - centerDegree):
                    minDegree = degrees_sorted_po[degrees_sorted_po.index(minDegree)-1]
                    for v2 in degrees_po[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                else:
                    maxDegree = degrees_sorted_po[degrees_sorted_po.index(maxDegree)+1]
                    for v2 in degrees_po[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
        except StopIteration:
            minDegree = degree_v[3]
            maxDegree = degree_v[3]
            centerDegree = degree_v[3]
            c_v = 0  
        try:
            while True:
                if (degrees_sorted_no.index(minDegree) == 0) and (degrees_sorted_no.index(maxDegree) < len(degrees_sorted_no) - 1) :
                    maxDegree = degrees_sorted_no[degrees_sorted_no.index(maxDegree)+1]
                    for v2 in degrees_no[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_no.index(minDegree) > 0) and (degrees_sorted_no.index(maxDegree) == len(degrees_sorted_no) - 1) :
                    minDegree = degrees_sorted_no[degrees_sorted_no.index(minDegree)-1]
                    for v2 in degrees_no[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                elif (degrees_sorted_no.index(minDegree) == 0) and (degrees_sorted_no.index(maxDegree) == len(degrees_sorted_no) - 1) :
                    raise StopIteration
                elif abs(degrees_sorted_no[degrees_sorted_no.index(minDegree)-1] - centerDegree) < abs(degrees_sorted_no[degrees_sorted_no.index(maxDegree)+1] - centerDegree):
                    minDegree = degrees_sorted_no[degrees_sorted_no.index(minDegree)-1]
                    for v2 in degrees_no[minDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
                else:
                    maxDegree = degrees_sorted_no[degrees_sorted_no.index(maxDegree)+1]
                    for v2 in degrees_no[maxDegree]:
                        if(v != v2):
                            vertices.append(v2)
                            c_v += 1
                        if(c_v > a_vertices_selected):
                            raise StopIteration
        except StopIteration:
            pass

    except StopIteration:
        #logging.info("Vértice {} - próximos selecionados.".format(v))
        return list(vertices)

    return list(vertices)


def get_vertices_4axis_directed_shrink(v,degree_v,degrees,a_vertices,mapping_degree2AD_ni,mapping_degree2AD_pi,mapping_degree2AD_no,mapping_degree2AD_po): # 
    '''
    degree_v: v's degree+i, v's degree-i, v's degree+o, v's degree-o quaternion style
    degrees: 4D matrix style
    a_vertices: # nodes
    !!! Note: var name different with outside!!!
    '''
    a_vertices_selected = int(2 * math.log(a_vertices,2) + 1)
    #logging.info("Selecionando {} próximos ao vértice {} ...".format(int(a_vertices_selected),v))
    vertices = deque()
    try:
        c_v = 0  
        if (c_v + len(degrees[degree_v]) <= a_vertices_selected):
            vertices += degrees[degree_v]
            c_v = len(degrees[degree_v])
        else:
            vertices += random.sample(degrees[degree_v], a_vertices_selected - c_v)
            c_v = a_vertices_selected
        #     raise StopIteration
        # if(c_v >= a_vertices_selected):
        #     raise StopIteration
        a_vertices_selected = 2*a_vertices_selected - c_v
        a_vertices_selected = int(a_vertices_selected//4 + 1)
        minDegree = degree_v[0]
        maxDegree = degree_v[0]
        centerDegree = degree_v[0]
        c_v = 0  
        try:
            while True:
                tmp_list = list(mapping_degree2AD_pi.keys())
                tmp_index = -1
                if (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) < len(mapping_degree2AD_pi) - 1) :
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                elif (tmp_list.index(minDegree) > 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_pi) - 1) :
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                elif (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_pi) - 1) :
                    raise StopIteration
                elif abs(tmp_list[tmp_list.index(minDegree)-1] - centerDegree) < abs(tmp_list[tmp_list.index(maxDegree)+1] - centerDegree):
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                else:
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                if (c_v + len(mapping_degree2AD_pi[tmp_index]) <= a_vertices_selected):
                    vertices += mapping_degree2AD_pi[tmp_index]
                    c_v += len(mapping_degree2AD_pi[tmp_index])
                else:
                    vertices += random.sample(mapping_degree2AD_pi[tmp_index], a_vertices_selected - c_v)
                    raise StopIteration
                if(c_v >= a_vertices_selected):
                    raise StopIteration
        except StopIteration:
            minDegree = degree_v[1]
            maxDegree = degree_v[1]
            centerDegree = degree_v[1]
            c_v = 0  
        try:
            while True:
                tmp_list = list(mapping_degree2AD_ni.keys())
                tmp_index = -1
                if (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) < len(mapping_degree2AD_ni) - 1) :
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                elif (tmp_list.index(minDegree) > 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_ni) - 1) :
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                elif (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_ni) - 1) :
                    raise StopIteration
                elif abs(tmp_list[tmp_list.index(minDegree)-1] - centerDegree) < abs(tmp_list[tmp_list.index(maxDegree)+1] - centerDegree):
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                else:
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                if (c_v + len(mapping_degree2AD_ni[tmp_index]) <= a_vertices_selected):
                    vertices += mapping_degree2AD_ni[tmp_index]
                    c_v += len(mapping_degree2AD_ni[tmp_index])
                else:
                    vertices += random.sample(mapping_degree2AD_ni[tmp_index], a_vertices_selected - c_v)
                    raise StopIteration
                if(c_v >= a_vertices_selected):
                    raise StopIteration
        except StopIteration:
            minDegree = degree_v[2]
            maxDegree = degree_v[2]
            centerDegree = degree_v[2]
            c_v = 0  
        try:
            while True:
                tmp_list = list(mapping_degree2AD_po.keys())
                tmp_index = -1
                if (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) < len(mapping_degree2AD_po) - 1) :
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                elif (tmp_list.index(minDegree) > 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_po) - 1) :
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                elif (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_po) - 1) :
                    raise StopIteration
                elif abs(tmp_list[tmp_list.index(minDegree)-1] - centerDegree) < abs(tmp_list[tmp_list.index(maxDegree)+1] - centerDegree):
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                else:
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                if (c_v + len(mapping_degree2AD_po[tmp_index]) <= a_vertices_selected):
                    vertices += mapping_degree2AD_po[tmp_index]
                    c_v += len(mapping_degree2AD_po[tmp_index])
                else:
                    vertices += random.sample(mapping_degree2AD_po[tmp_index], a_vertices_selected - c_v)
                    raise StopIteration
                if(c_v >= a_vertices_selected):
                    raise StopIteration
        except StopIteration:
            minDegree = degree_v[3]
            maxDegree = degree_v[3]
            centerDegree = degree_v[3]
            c_v = 0  
        try:
            while True:
                tmp_list = list(mapping_degree2AD_no.keys())
                tmp_index = -1
                if (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) < len(mapping_degree2AD_no) - 1) :
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                elif (tmp_list.index(minDegree) > 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_no) - 1) :
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                elif (tmp_list.index(minDegree) == 0) and (tmp_list.index(maxDegree) == len(mapping_degree2AD_no) - 1) :
                    raise StopIteration
                elif abs(tmp_list[tmp_list.index(minDegree)-1] - centerDegree) < abs(tmp_list[tmp_list.index(maxDegree)+1] - centerDegree):
                    minDegree = tmp_list[tmp_list.index(minDegree)-1]
                    tmp_index = minDegree
                else:
                    maxDegree = tmp_list[tmp_list.index(maxDegree)+1]
                    tmp_index = maxDegree
                if (c_v + len(mapping_degree2AD_no[tmp_index]) <= a_vertices_selected):
                    vertices += mapping_degree2AD_no[tmp_index]
                    c_v += len(mapping_degree2AD_no[tmp_index])
                else:
                    vertices += random.sample(mapping_degree2AD_no[tmp_index], a_vertices_selected - c_v)
                    raise StopIteration
                if(c_v >= a_vertices_selected):
                    raise StopIteration
        except StopIteration:
            pass

    except StopIteration:
        #logging.info("Vértice {} - próximos selecionados.".format(v))
        return list(vertices)

    return list(vertices)

def splitDegreeList_complex_directed(part,c,Gpi,Gni,Gpo,Gno,compactDegree):
    if(compactDegree):
        logging.info("Recovering compactDegreeList from disk...")
        degreeList = restoreVariableFromDisk('compactDegreeList')
    else:
        logging.info("Recovering degreeList from disk...")
        degreeList = restoreVariableFromDisk('degreeList')

    logging.info("Recovering degree vector from disk...")
    degrees = restoreVariableFromDisk('degrees_vector')
    degrees_sorted_ni = restoreVariableFromDisk('degrees_vector_negativeInList')
    degrees_sorted_pi = restoreVariableFromDisk('degrees_vector_positiveInList')
    degrees_sorted_no = restoreVariableFromDisk('degrees_vector_negativeOutList')
    degrees_sorted_po = restoreVariableFromDisk('degrees_vector_positiveOutList')
    degrees_ni = restoreVariableFromDisk('degrees_vector_node_negativeInList')
    degrees_pi = restoreVariableFromDisk('degrees_vector_node_positiveInList')
    degrees_no = restoreVariableFromDisk('degrees_vector_node_negativeOutList')
    degrees_po = restoreVariableFromDisk('degrees_vector_node_positiveOutList')
    degreeListsSelected = {}
    vertices = {}
    a_vertices = len(list(set(list(Gpi.keys()) + list(Gni.keys()) + list(Gpo.keys()) + list(Gno.keys()))))
    for v in c:
        # nbs = get_vertices_4axis_directed(v,np.quaternion(len(Gpi[v]), len(Gni[v]), len(Gpo[v]), len(Gno[v])),degrees,a_vertices,degrees_sorted_ni,degrees_sorted_pi,degrees_sorted_no,degrees_sorted_po,degrees_ni,degrees_pi,degrees_no,degrees_po) # 
        nbs = get_vertices_4axis_directed(v,(len(Gpi[v]), len(Gni[v]), len(Gpo[v]), len(Gno[v])),degrees,a_vertices,degrees_sorted_ni,degrees_sorted_pi,degrees_sorted_no,degrees_sorted_po,degrees_ni,degrees_pi,degrees_no,degrees_po) # 
        # print (v, nbs)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    saveVariableOnDisk(vertices,'split-vertices-'+str(part))
    saveVariableOnDisk(degreeListsSelected,'split-degreeList-'+str(part))

def splitDegreeList_complex_directed_shrink(part,c,Gpi,Gni,Gpo,Gno,compactDegree):
    if(compactDegree):
        logging.info("Recovering compactDegreeList from disk...")
        degreeList = restoreVariableFromDisk('compactDegreeList')
    else:
        logging.info("Recovering degreeList from disk...")
        degreeList = restoreVariableFromDisk('degreeList')

    logging.info("Recovering degree vector from disk...")
    degrees = restoreVariableFromDisk('degrees_vector')
    mapping_degree2AD_ni = restoreVariableFromDisk('degrees_vector_negativeInList')
    mapping_degree2AD_pi = restoreVariableFromDisk('degrees_vector_positiveInList')
    mapping_degree2AD_no = restoreVariableFromDisk('degrees_vector_negativeOutList')
    mapping_degree2AD_po = restoreVariableFromDisk('degrees_vector_positiveOutList')
    mapping_AD2node_ni = restoreVariableFromDisk('degrees_vector_node_negativeInList')
    mapping_AD2node_pi = restoreVariableFromDisk('degrees_vector_node_positiveInList')
    mapping_AD2node_no = restoreVariableFromDisk('degrees_vector_node_negativeOutList')
    mapping_AD2node_po = restoreVariableFromDisk('degrees_vector_node_positiveOutList')
    degreeListsSelected = {}
    vertices = {}
    a_vertices = len(list(set(list(Gpi.keys()) + list(Gni.keys()) + list(Gpo.keys()) + list(Gno.keys()))))
    for v in c:
        # nbs = get_vertices_4axis_directed(v,np.quaternion(len(Gpi[v]), len(Gni[v]), len(Gpo[v]), len(Gno[v])),degrees,a_vertices,degrees_sorted_ni,degrees_sorted_pi,degrees_sorted_no,degrees_sorted_po,degrees_ni,degrees_pi,degrees_no,degrees_po) # 
        nbs = get_vertices_4axis_directed_shrink(v,(mapping_degree2AD_pi[len(Gpi[v])], mapping_degree2AD_ni[len(Gni[v])], mapping_degree2AD_po[len(Gpo[v])], mapping_degree2AD_no[len(Gno[v])]),degrees,a_vertices,mapping_AD2node_ni,mapping_AD2node_pi,mapping_AD2node_no,mapping_AD2node_po)
                
        # print (v, nbs)
        vertices[v] = nbs
        degreeListsSelected[v] = degreeList[v]
        for n in nbs:
            degreeListsSelected[n] = degreeList[n]

    saveVariableOnDisk(vertices,'split-vertices-'+str(part))
    saveVariableOnDisk(degreeListsSelected,'split-degreeList-'+str(part))

def calc_distances_complex_directed(part, compactDegree = False, scale_free = True):
    # scale_free = False
    # scale_free = True
    # print (part)
    vertices = restoreVariableFromDisk('split-vertices-'+str(part))
    degreeList = restoreVariableFromDisk('split-degreeList-'+str(part))
    # print (part)

    distances = {}

    if compactDegree:
        dist_func = cost_complex_vector_Euclidean_logscale_max_directed if scale_free else cost_complex_vector_Euclidean_max_directed
        for v1,nbs in vertices.items():
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z),  (x[1], 0, 0, 0)) for x in value_]})
            for v2 in nbs:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    # distances[v1,v2][layer] = dist 
                    # distances[v1,v2][layer] = np.exp(dist)**np.e
    else:
        dist_func = cost_complex_vector_Euclidean_logscale_directed if scale_free else cost_complex_vector_Euclidean_directed
        # dist_func = cost_complex_vector_2area_logscale
        # dist_func = cost_complex_sinusoidalWave_logscale_vector
        for v1,nbs in vertices.items():
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})
            for v2 in nbs:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    # distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    distances[v1,v2][layer] = dist 
                    # distances[v1,v2][layer] = np.exp(dist)**np.e

#     if compactDegree:
#         dist_func = cost_max
#     else:
#         dist_func = cost

#     for v1,nbs in vertices.items():
#         lists_v1 = degreeList[v1]

#         for v2 in nbs:
#             t00 = time()
#             lists_v2 = degreeList[v2]

#             max_layer = min(len(lists_v1),len(lists_v2))
#             distances[v1,v2] = {}

#             for layer in range(0,max_layer):
#                 dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
# # start dist_minus calculation

# #  end  dist_minus calculation
#                 distances[v1,v2][layer] = dist

            # t11 = time()
            # logging.info('fastDTW between vertices ({}, {}). Time: {}s'.format(v1,v2,(t11-t00)))


    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def calc_distances_all_complex_directed(vertices,list_vertices,degreeList,part, compactDegree = False, scale_free = True):
    # scale_free = False
    # scale_free = True
    distances = {}
    cont = 0

    # TODO
    if compactDegree:
        dist_func = cost_complex_vector_Euclidean_logscale_max_directed if scale_free else cost_complex_vector_Euclidean_max_directed
        for v1 in vertices:
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})
            for v2 in list_vertices[cont]:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[((x[0].w, x[0].x, x[0].y, x[0].z), (x[1], 0, 0, 0)) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    # print (v1,v2, layer, np.exp(dist))
                    # distances[v1,v2][layer] = dist 
                    # distances[v1,v2][layer] = np.exp(dist)**np.e


            cont += 1

    else:
        dist_func = cost_complex_vector_Euclidean_logscale_directed if scale_free else cost_complex_vector_Euclidean_directed
        # dist_func = cost_complex_vector_2area_logscale
        # dist_func = cost_complex_sinusoidalWave_logscale_vector
        for v1 in vertices:
            lists_v1 = degreeList[v1]
            lists_v1_new = {}
            for key_,value_ in lists_v1.items():
                lists_v1_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})
            for v2 in list_vertices[cont]:
                lists_v2 = degreeList[v2]
                lists_v2_new = {}
                for key_,value_ in lists_v2.items():
                    lists_v2_new.update({key_:[(x.w, x.x, x.y, x.z) for x in value_]})

                max_layer = min(len(lists_v1),len(lists_v2))
                distances[v1,v2] = {}
                for layer in range(0,max_layer):
                    #t0 = time()
                    
                    # print ('101 111')
                    # print (layer,lists_v1_new[layer],lists_v2_new[layer])
                    dist, path = fastdtw(lists_v1_new[layer],lists_v2_new[layer],radius=1,dist=dist_func)
                    # print ('100 111')
                    # dist, path = fastdtw(lists_v1[layer],lists_v2[layer],radius=1,dist=dist_func)
    # start dist_minus calculation

    #  end  dist_minus calculation
                    #t1 = time()
                    #logging.info('D ({} , {}), Tempo fastDTW da camada {} : {}s . Distância: {}'.format(v1,v2,layer,(t1-t0),dist))    
                    # distances[v1,v2][layer] = np.exp(dist) # Link to # edges might be better!!! TODO
                    distances[v1,v2][layer] = np.exp(dist)
                    # print (v1,v2, layer, np.exp(dist), dist)
                    # distances[v1,v2][layer] = np.exp(dist)**np.e


            cont += 1

    preprocess_consolides_distances(distances)
    saveVariableOnDisk(distances,'distances-'+str(part))
    return

def preprocess_consolides_distances(distances, startLayer = 1):

    logging.info('Consolidating distances...')

    for vertices,layers in distances.items():
        keys_layers = sorted(list(layers.keys()))
        startLayer = min(len(keys_layers),startLayer)
        for layer in range(0,startLayer):
            keys_layers.pop(0)


        for layer in keys_layers:
            layers[layer] += layers[layer - 1]

    logging.info('Distances consolidated.')

def exec_bfs_compact_complex_directed(Gpi,Gni,Gpo,Gno,workers,calcUntilLayer,opt4):
    
    futures = {}
    degreeList = {}

    t0 = time()
    vertices = list(sorted(set(list(Gpi.keys()) + list(Gni.keys()) + list(Gpo.keys()) + list(Gno.keys()))))
    parts = workers
    chunks = partition(vertices,parts)

    # logging.info('Capturing larger degree...')
    # for v in vertices:
    #     if(len(Gpi[v])+len(Gni[v])+len(Gpo[v])+len(Gno[v]) > maxDegree):
    #         maxDegree = len(Gpi[v])+len(Gni[v])+len(Gpo[v])+len(Gno[v])
    # logging.info('Larger degree captured')

    # part = 1
    
    # print (chunks) #DEBUG
    # for c in chunks:
    #     dl = getCompactDegreeListsVertices_complex_directed(Gpi,Gni,Gpo,Gno,c,maxDegree,calcUntilLayer)
    #     degreeList.update(dl)
    # print (degreeList) #DEBUG
    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(getCompactDegreeListsVertices_complex_directed,Gpi,Gni,Gpo,Gno,c,calcUntilLayer,opt4)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)

    logging.info("Saving degreeList on disk...")
    saveVariableOnDisk(degreeList,'compactDegreeList')
    t1 = time()
    logging.info('Execution time - BFS: {}m'.format((t1-t0)/60))
    return

def exec_bfs_complex_directed(Gpi,Gni,Gpo,Gno,workers,calcUntilLayer):
    futures = {}
    degreeList = {}

    t0 = time()
    vertices = list(sorted(set(list(Gpi.keys()) + list(Gni.keys()) + list(Gpo.keys()) + list(Gno.keys()))))
    parts = workers
    chunks = partition(vertices,parts)
    # for c in chunks:
    #     dl = getDegreeListsVertices_complex_directed(Gpi,Gni,Gpo,Gno,c,calcUntilLayer)
    #     degreeList.update(dl)

    with ProcessPoolExecutor(max_workers=workers) as executor:

        part = 1
        for c in chunks:
            job = executor.submit(getDegreeListsVertices_complex_directed,Gpi,Gni,Gpo,Gno,c,calcUntilLayer)
            futures[job] = part
            part += 1

        for job in as_completed(futures):
            dl = job.result()
            v = futures[job]
            degreeList.update(dl)

    logging.info("Saving degreeList on disk...")
    saveVariableOnDisk(degreeList,'degreeList')
    t1 = time()
    logging.info('Execution time - BFS: {}m'.format((t1-t0)/60))

    return

def generate_distances_network_part1(workers):
    parts = workers
    weights_distances = {}
    for part in range(1,parts + 1):    

        logging.info('Executing part {}...'.format(part))
        distances = restoreVariableFromDisk('distances-'+str(part))
        for vertices,layers in distances.items():
            for layer,distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in weights_distances):
                    weights_distances[layer] = {}
                weights_distances[layer][vx,vy] = distance

        logging.info('Part {} executed.'.format(part))

    for layer,values in weights_distances.items():
        saveVariableOnDisk(values,'weights_distances-layer-'+str(layer))
    return

def generate_distances_network_part2(workers):
    parts = workers
    graphs = {}
    for part in range(1,parts + 1):

        logging.info('Executing part {}...'.format(part))
        distances = restoreVariableFromDisk('distances-'+str(part))

        for vertices,layers in distances.items():
            for layer,distance in layers.items():
                vx = vertices[0]
                vy = vertices[1]
                if(layer not in graphs):
                    graphs[layer] = {}
                if(vx not in graphs[layer]):
                   graphs[layer][vx] = [] 
                if(vy not in graphs[layer]):
                   graphs[layer][vy] = [] 
                graphs[layer][vx].append(vy)
                graphs[layer][vy].append(vx)
        logging.info('Part {} executed.'.format(part))

    for layer,values in graphs.items():
        saveVariableOnDisk(values,'graphs-layer-'+str(layer))

    return

def generate_distances_network_part3():

    layer = 0
    while(isPickle('graphs-layer-'+str(layer))):
        graphs = restoreVariableFromDisk('graphs-layer-'+str(layer))
        weights_distances = restoreVariableFromDisk('weights_distances-layer-'+str(layer))

        logging.info('Executing layer {}...'.format(layer))
        alias_method_j = {}
        alias_method_q = {}
        weights = {}

        for v,neighbors in graphs.items():
            e_list = deque()
            sum_w = 0.0


            for n in neighbors:
                if (v,n) in weights_distances:
                    wd = weights_distances[v,n]
                else:
                    wd = weights_distances[n,v]
                w = np.exp(-float(wd))
                e_list.append(w)
                sum_w += w

            e_list = [x / sum_w for x in e_list]
            weights[v] = e_list
            J, q = alias_setup(e_list)
            alias_method_j[v] = J
            alias_method_q[v] = q

        saveVariableOnDisk(weights,'distances_nets_weights-layer-'+str(layer))
        saveVariableOnDisk(alias_method_j,'alias_method_j-layer-'+str(layer))
        saveVariableOnDisk(alias_method_q,'alias_method_q-layer-'+str(layer))
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info('Weights created.')

    return


def generate_distances_network_part4():
    logging.info('Consolidating graphs...')
    graphs_c = {}
    layer = 0
    while(isPickle('graphs-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))
        graphs = restoreVariableFromDisk('graphs-layer-'+str(layer))
        graphs_c[layer] = graphs
        logging.info('Layer {} executed.'.format(layer))
        layer += 1


    logging.info("Saving distancesNets on disk...")
    saveVariableOnDisk(graphs_c,'distances_nets_graphs')
    logging.info('Graphs consolidated.')
    return

def generate_distances_network_part5():
    alias_method_j_c = {}
    layer = 0
    while(isPickle('alias_method_j-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))          
        alias_method_j = restoreVariableFromDisk('alias_method_j-layer-'+str(layer))
        alias_method_j_c[layer] = alias_method_j
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_j on disk...")
    saveVariableOnDisk(alias_method_j_c,'nets_weights_alias_method_j')

    return

def generate_distances_network_part6():
    alias_method_q_c = {}
    layer = 0
    while(isPickle('alias_method_q-layer-'+str(layer))):
        logging.info('Executing layer {}...'.format(layer))          
        alias_method_q = restoreVariableFromDisk('alias_method_q-layer-'+str(layer))
        alias_method_q_c[layer] = alias_method_q
        logging.info('Layer {} executed.'.format(layer))
        layer += 1

    logging.info("Saving nets_weights_alias_method_q on disk...")
    saveVariableOnDisk(alias_method_q_c,'nets_weights_alias_method_q')

    return

def generate_distances_network(workers):
    t0 = time()
    logging.info('Creating distance network...')

    os.system("rm "+returnPathPickles()+"weights_distances-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part1,workers)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 1: {}s'.format(t))

    t0 = time()
    os.system("rm "+returnPathPickles()+"graphs-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part2,workers)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 2: {}s'.format(t))
    logging.info('distance network created.')

    logging.info('Transforming distances into weights...')

    t0 = time()
    os.system("rm "+returnPathPickles()+"distances_nets_weights-layer-*.pickle")
    os.system("rm "+returnPathPickles()+"alias_method_j-layer-*.pickle")
    os.system("rm "+returnPathPickles()+"alias_method_q-layer-*.pickle")
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part3)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 3: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part4)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 4: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part5)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 5: {}s'.format(t))

    t0 = time()
    with ProcessPoolExecutor(max_workers=1) as executor:
        job = executor.submit(generate_distances_network_part6)
        job.result()
    t1 = time()
    t = t1-t0
    logging.info('- Time - part 6: {}s'.format(t))

    return

def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q

def merge_distance_by_sech(height):
    '''
    $$f(x,y)=(x+y)^2+ h*\frac{cosh^2(\sqrt{2}|\frac{x-y}{2}|) - sinh^2(\sqrt{2}|\frac{x-y}{2}|))}{cosh^2(\sqrt{2}|\frac{x-y}{2}|)}$$
    '''
    return lambda x,y: (x+y+1)**2 + height*(np.cosh(np.sqrt(2)*abs(x-y)/2)**2 - np.sinh(np.sqrt(2)*abs(x-y)/2)**2)/np.cosh(np.sqrt(2)*abs(x-y)/2)**2

def mylambda():
    return []