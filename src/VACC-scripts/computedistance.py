# computedistance.py
#
# using a distance metric, find the distance of all of the stories
# 
# USAGE
#
# python computedistance.py [option start num] [option number to do]

import codecs # handle utf8
import sys

logfile = "analysislogactually.txt"
# logfile = "analysislogactually.txt.test"

def findbooknum(count,num=1):
    f = open(logfile,"r")
    for i in xrange(count):
        f.readline()
    books = []
    for i in xrange(num):
        books.append(f.readline().rstrip().split('\t')[0])
    return books

def distance(a,b):
    return sum([abs(a[i]-b[i]) for i in xrange(len(a))])

def processbook(booknum):
    # open it's timeseries
    f = open("timeseries/"+booknum+".csv","r")
    a = map(int,f.read().rstrip().split(","))
    f.close()
    # print a
    
    # look it up in the analysis log
    f = open(logfile,"r")
    distances = []
    for line in f:
        bookinfo = line.rstrip().split('\t')
        g = open("timeseries/"+bookinfo[0]+".csv","r")
        b = map(int,g.read().rstrip().split(","))
        # print b
        g.close()
        d = distance(a,b)
        # print d
        distances.append(d)
    f.close()

    # print distances
    f = open("sumdistance/"+booknum+".csv","w")
    f.write("{0:.0f}".format(distances[0]))
    for d in distances[1:]:
        f.write("\n{0:.0f}".format(d))
    f.close()
    
if __name__ == "__main__":
    count = int(sys.argv[1])
    if len(sys.argv) > 2:
        numtoread = int(sys.argv[2])
        booknums = findbooknum(count,num=numtoread)
    else:
        booknums = findbooknum(count)

    for booknum in booknums:
        processbook(booknum)

    print "complete"








