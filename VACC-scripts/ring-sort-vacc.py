from os import listdir
from os.path import isfile, join
import sys
from json import loads
from re import findall,UNICODE
import numpy as np
import pickle

from datetime import datetime

use_cache = True

def cityBlock(a,b):
    return np.sum(np.abs(a-b))

def costFun5(distanceMatrix,weightMatrix,i,swap,oldcost):
    # compute the dot products for the old columns
    oldcol = np.dot(distanceMatrix[:,i],weightMatrix[:,i])+np.dot(distanceMatrix[:,swap],weightMatrix[:,swap])
    # compute the dot products for the old rows
    oldrow = np.dot(distanceMatrix[i,:],weightMatrix[i,:])+np.dot(distanceMatrix[swap,:],weightMatrix[swap,:])
    # compute the dot products for the new columns
    newcol = np.dot(distanceMatrix[:,swap],weightMatrix[:,i])+np.dot(distanceMatrix[:,i],weightMatrix[:,swap])
    # compute the dot products for the new rows
    newrow = np.dot(distanceMatrix[swap,:],weightMatrix[i,:])+np.dot(distanceMatrix[i,:],weightMatrix[swap,:])
    return oldcost-oldcol-oldrow+newcol+newrow

def reshaper(newShape,matrix):
    newMatrix = np.zeros(matrix.shape)
    for i in range(matrix.shape[0]):
        newMatrix[i,:] = matrix[i,newShape]
    newMatrix = newMatrix[newShape,:]
    return newMatrix

def computeBestSwap(distanceMatrix,n,i,weightMatrix,oldcost,v=False):
    maxDistanceReduction = 0
    bestSwap = 0
    if n < distanceMatrix.shape[0]:
        iterList = np.random.randint(0,high=distanceMatrix.shape[0],size=n)
    else:
        iterList = range(distanceMatrix.shape[0])
    for swap in iterList:
        # a = np.arange(distanceMatrix.shape[0])
        # a[[i,swap]] = [swap,i]
        # tmpswapped = distanceMatrix[a,a]
        # b = np.sum(np.multiply(distanceMatrix[np.ix_(a,a)],weightMatrix),axis=(0,1))
        # b = costFun3(distanceMatrix[np.ix_(a,a)],weightMatrix)
        b = costFun5(distanceMatrix,weightMatrix,i,swap,oldcost)
        if oldcost-b > maxDistanceReduction:
            # print("found a swap that reduces distance to: {0}".format(b))
            maxDistanceReduction = oldcost-b
            bestSwap = swap
    if v:
        print("best swap: distance\tnode")
        print("           {0:.3f}\t{1}".format(maxDistanceReduction,bestSwap))

    return (maxDistanceReduction,bestSwap)

def runSortingAlgorithm(sortedDistances,sorted_index,exponent=-.5,niter=0,v=False,plot_every=0):
    weightMatrix = np.zeros(sortedDistances.shape)
    for i in range(weightMatrix.shape[0]):
        weightMatrix[i,:] = np.abs(np.arange(weightMatrix.shape[1])-i)
        # this index doesn't matter! (but set it to 1 so power doesn't complain)
        weightMatrix[i,i] = 1.
        weightMatrix[i,:] = np.power(weightMatrix[i,:],exponent)

    # sorted_index = np.arange(sortedDistances.shape[0])
    if niter == 0:
        niter = sortedDistances.shape[0]
    # niter = 10
    n = sortedDistances.shape[0]
    # n = np.floor(np.sqrt(sortedDistances.shape[0]))
    # n = 10
    allcost = np.zeros(niter+1)
    allcost[0] = np.sum(np.multiply(sortedDistances,weightMatrix),axis=(0,1))
    if v:
        print("initial cost: {0:.2f}".format(allcost[0]))
    for i in range(niter):
        # so we can run more iterations than the number of books
        node = np.mod(i,sortedDistances.shape[0])
        if v:
            print("iteration {0}, node {1}".format(i,node))
        # find the best swap
        maxD,bswap = computeBestSwap(sortedDistances,n,node,weightMatrix,allcost[i],v=v)
        # make that swap
        if maxD > 0:
            if v:
                print("swapping")
            a = np.arange(sortedDistances.shape[0])
            a[[node,bswap]] = [bswap,node]
            sorted_index[[node,bswap]] = sorted_index[[bswap,node]]
            sortedDistances = sortedDistances[np.ix_(a,a)]
            # save the next cost function
            allcost[i+1] = np.sum(np.multiply(sortedDistances,weightMatrix),axis=(0,1))
        else:
            if v:
                print("no better position found, not swapping")
            allcost[i+1] = allcost[i]
            # break
        if i > sortedDistances.shape[0] and allcost[i-sortedDistances.shape[0]] == allcost[i]:
            print("no improvement over the entire last sweep!")
            break
        if v:
            print("new cost: {0:.2f}".format(allcost[i+1]))
        if plot_every > 0:
            if i%plot_every == 0:
                plotDistanceMatrix(allDistancesCentered,sortedDistances,suffix=i,cap=50)
        
    if v:
        print("saving to a file...")
    return sortedDistances,allcost,sorted_index
        
if __name__ == "__main__":
    if isfile("data/gutenberg/pairwise-distance-mean0-matrix-cache.p"):
        try:
            allDistancesCentered = pickle.load(open("data/gutenberg/pairwise-distance-mean0-matrix-cache.p","rb"))
        except:
            if isfile("data/gutenberg/pairwise-distance-mean0-matrix-cache.csv"):
                # this saves it as a csv, for transport
                # np.savetxt("data/gutenberg/pairwise-distance-mean0-matrix-cache.csv",allDistancesCentered,delimiter=",",fmt="%.18e")
                allDistancesCentered = np.genfromtxt("data/gutenberg/pairwise-distance-mean0-matrix-cache.csv")
            else:
                raise("pickle failed, csv missing: rerun the notebook to generate the distance matrix csv")
    else:
        raise("pickle missing: rerun the notebook to generate the distance matrix pickle")

    # big_matrix = pickle.load(open("data/gutenberg/timeseries-matrix-cache.p","rb"))
    # big_matrix_mean0 = big_matrix-np.tile(big_matrix.mean(axis=1),(200,1)).transpose()

    if sys.argv[1] == "optimize":
        # don't have to give it a num ops
        if len(sys.argv) > 2:
            max_iterations = int(sys.argv[2])
        else:
            max_iterations = 1000
            
        # don't have to name it
        if len(sys.argv) > 3:
            name = sys.argv[3]
        else:
            name = ""

        # randomly shuffle and save
        # if there is a name, use that. otherwise, time stamp
        if len(name) > 0:
            formatted_fname = "data/gutenberg/{0}-allDistances-cityblock-shuffled-{1}.p".format(name,max_iterations)
        else:
            now = datetime.now()
            formatted_fname = now.strftime("data/gutenberg/%Y-%m-%d-%H-%M-allDistances-cityblock-shuffled-{}.p".format(max_iterations))

        # shuffle it...
        a = np.arange(allDistancesCentered.shape[0])
        np.random.shuffle(a)
        allDistancesCenteredShuffled = allDistancesCentered[np.ix_(a,a)]
        # pickle.dump(allDistancesCenteredShuffled,open(formatted_fname,"wb"),pickle.HIGHEST_PROTOCOL)
        # open(formatted_fname,"w").write("\n".join(map(str,a)))
        np.savetxt(formatted_fname.replace(".p",".csv"),a,delimiter="\n",fmt="%.0f")
        
        # sortedDistances,cost = runSortingAlgorithm(allDistancesCenteredShuffled,niter=max_iterations,v=True)
        sortedDistances,cost,sorted_index = runSortingAlgorithm(allDistancesCenteredShuffled,a,niter=max_iterations,v=True)
                

        if len(name) > 0:
            formatted_fname = "data/gutenberg/{0}-allDistances-cityblock-sorted-{1}.p".format(name,max_iterations)
            formatted_fname_cost = "data/gutenberg/{0}-cost-cityblock-sorted-{1}.p".format(name,max_iterations)
        else:
            formatted_fname = now.strftime("data/gutenberg/%Y-%m-%d-%H-%M-allDistances-cityblock-sorted-{}.p".format(max_iterations))
            formatted_fname_cost = now.strftime("data/gutenberg/%Y-%m-%d-%H-%M-cost-cityblock-sorted-{}.p".format(max_iterations))


        # pickle.dump(sortedDistances,open(formatted_fname,"wb"),pickle.HIGHEST_PROTOCOL)
        np.savetxt(formatted_fname.replace(".p",".csv"),sorted_index,delimiter="\n",fmt="%.0f")
        # pickle.dump(cost,open(formatted_fname_cost,"wb"),pickle.HIGHEST_PROTOCOL)
        np.savetxt(formatted_fname_cost.replace(".p",".csv"),cost,delimiter="\n",fmt="%.3f")
