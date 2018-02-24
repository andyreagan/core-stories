from os import listdir
from os.path import isfile, join
import sys
# local
sys.path.append("/Users/andyreagan/tools/python")
from kitchentable.dogtoys import *
from json import loads
from re import findall,UNICODE
from labMTsimple.labMTsimple.speedy import LabMT
my_LabMT = LabMT()
from labMTsimple.labMTsimple.storyLab import *
import numpy as np
from database.bookclass import Book_raw_data
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

def plotDistanceMatrix(data,data_sorted,suffix='',useclim=True,cap=120,figsize=30,cmap="magma",name="",niter=""):
    '''Plot the saved matrix files, and the stories that are given.'''
    
    # don't load the plotting stuff unless it's needed, which is here
    
    # all our essentials
    from matplotlib import rc,rcParams
    import matplotlib.patches as patches
    
    rc('font', family='serif')
    rc('font', family='cmr10')
    rc('text', usetex='true') 
    
    rcParams.update({'font.size': 12})
    import matplotlib.pyplot as plt
    
    # numel = len(q)-1
    numel = data.shape[0]

    fig = plt.figure(figsize=(22.5,7.5))
    ax1 = fig.add_axes([0.15,0.15,0.8/3,0.8-.04])
    # screw memory...I have a SSD!
    # plt.subplot(1,2,1)
    data_capped = data.copy()
    data_capped[data > cap] = cap
    ax1.imshow(data_capped,cmap=plt.get_cmap(cmap))
    # # ax1.colorbar()
    # # plt.colorbar(cax,shrink=0.8)
    # # plt.clim([0,120])
    # plt.title('unsorted distance matrix')
    ax1.set_xlabel('Book ID')
    # plt.ylabel('book num')

    data_sorted_capped = data_sorted.copy()
    data_sorted_capped[data_sorted > cap] = cap
    ax2 = fig.add_axes([0.15+0.8/3,0.15,0.8/3,0.8-.04])
    ax2.imshow(data_sorted_capped,cmap=plt.get_cmap(cmap))
    # # ax1.colorbar()
    # # plt.colorbar(cax,shrink=0.8)
    # # plt.clim([0,120])
    # plt.title('unsorted distance matrix')
    ax2.set_xlabel('Book ID')
    ax2.set_yticklabels([])

    # go and find the shift! put the story with the greatest distance between neighbors in the
    # upper left
    shift = list(map(lambda x: x%data.shape[0],np.arange(data_sorted.shape[0])+np.argmax(data_sorted.sum(axis=1))))
    ax3 = fig.add_axes([0.15+0.8/3+0.8/3,0.15,0.8/3,0.8-.04])
    ax3.imshow(data_sorted_capped[shift,:][:,shift],cmap=plt.get_cmap(cmap))
    
    groups = [(550,100),(715,70),(950,50),(1020,110)]
    groups_fill = [(groups[i][0]+groups[i][1],groups[i+1][0]-groups[i][0]-groups[i][1]) for i in range(len(groups)-1)]
    for i in range(len(groups)):
        ax3.add_patch(patches.Rectangle(
                   (groups[i][0],groups[i][0]), groups[i][1], groups[i][1],
                   facecolor="none",
                   edgecolor="white",)
                 )
        ax3.text(groups[i][0]+groups[i][1]+20,groups[i][0]-20,
                 letters[i],
                 color="white",
                 fontsize=10,
                 ha="left",
                 va="top",)
    for i in range(len(groups_fill)):
        ax3.add_patch(patches.Rectangle(
                   (groups_fill[i][0],groups_fill[i][0]), groups_fill[i][1], groups_fill[i][1],
                   facecolor="none",
                   edgecolor="white",)
                 )
    # # ax1.colorbar()
    # # plt.colorbar(cax,shrink=0.8)
    # # plt.clim([0,120])
    # plt.title('unsorted distance matrix')
    ax3.set_xlabel('Book ID')
    ax3.set_yticklabels([])

    ax4 = fig.add_axes([0.15+0.8/3+0.8/3+0.8/3,0.15,0.03,0.8-.04])
    my_cmap = np.tile(np.arange(data.shape[0]),(100,1)).transpose()
    print(my_cmap)
    ax4.imshow(my_cmap,cmap=plt.get_cmap(cmap),origin="lower")
    ax4.set_xticks([])
    ax4.set_xticklabels([])
    ax4.yaxis.tick_right()
    ax4.set_yticks([0,numel*.25,numel*.5,numel*.75,numel])
    ax4.set_yticklabels(map(int,[0,cap*.25,cap*.5,cap*.75,cap]))
    
    print("saving...")
    mysavefig("sorted-matrix-{}-{}-n{}-i{}-.pdf".format(cap,suffix,name,niter),openfig=True,date=True,folder="media/figures/ring-sort")

    # groups = [(550,100),(715,70),(950,50),(1020,110)]
    # now let's go and plot the stories from each group with the average
    # and the most central ones to the groups
    for g in groups:
        sort_index
        
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
        
    elif sys.argv[1] == "plot":
        # need to give it a name to plot from
        name = sys.argv[3]
        # ^ this could be a date!
        max_iterations = sys.argv[2]

        cap = 120
        if len(sys.argv) > 4:
            cap = float(sys.argv[4])
        
        data = pickle.load(open("data/gutenberg/{0}-allDistances-cityblock-shuffled-{1}.p".format(name,max_iterations),"rb"))
        data_sorted = pickle.load(open("data/gutenberg/{0}-allDistances-cityblock-sorted-{1}.p".format(name,max_iterations),"rb"))

        # # can I recover the sorted indices?
        print(data[0,:])
        print(data_sorted[0,:])
        sorted_indices = [np.nonzero(data_sorted==data[i,:]) for i in range(data.shape[0])]
        print(sorted_indices[:10])
        # the answer is no, apparently
        
        # big_matrix = pickle.load(open("data/gutenberg/timeseries-matrix-cache.p","rb"))
        # big_matrix_mean0 = big_matrix-np.tile(big_matrix.mean(axis=1),(200,1)).transpose()
        
        # plotDistanceMatrix(data,data_sorted,name=name,niter=max_iterations,cap=cap)

