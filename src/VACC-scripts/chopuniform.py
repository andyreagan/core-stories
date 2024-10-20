# chopuniform.py
#
# chop up a text into a timeseries
# 
# USAGE
#
# python chopuniform.py [option start num]

import codecs # handle utf8
import re
import sys
from labMTsimple.storyLab import *
import os
from math import floor

def binn(somelist,value):
    # return the index of the value in the list (monotonic) for which
    # the given value is not greater than
    if value == somelist[0]:
        return 1
    else:
        index = 0
        while value > somelist[index]:
            index += 1
        return index

def coursegrain(t,points=21):
    # take a vector of scores
    # and break it down into a series of points
    extent = [min(t),max(t)]
    print extent
    # create the bins
    nbins = float(points)
    nbinsize = (extent[1]-extent[0])/nbins
    print nbinsize
    binsize = nbinsize + nbinsize*2/nbins
    print binsize
    bins = [extent[0]+i*binsize-nbinsize*2/nbins for i in xrange(points)]
    print bins
    # print bins
    # newvec = [binn(bins,v) for v in t]
    # print newvec
    
    # normalize starting point to 0
    tmpb = [binn(bins,v) for v in t]
    # return tmpb
    return [x-tmpb[0] for x in tmpb]

def chopper(words,labMT,labMTvector,labMTwordList,outfile,minSize=10000,numPoints=100):
    print "now splitting the text into chunks of size 10000"
    # print "and printing those frequency vectors"

    # initialize timeseries, only thing we're after
    timeseries = [0 for i in xrange(numPoints)]

    # how much to jump
    step = int(floor((len(words)-minSize)/(numPoints-1)))
    print "there are "+str(len(words))+" words in the book"
    print "step size "+str(step)

    # do it 99 times
    for i in xrange(numPoints-1):
        chunk = unicode(' ').join(words[(i*step):(minSize+i*step)])
        textValence,textFvec = emotion(chunk,labMT,shift=True,happsList=labMTvector)
        stoppedVec = stopper(textFvec,labMTvector,labMTwordList,stopVal=2.0)
        timeseries[i] = emotionV(stoppedVec,labMTvector)

    # final chunk
    i = numPoints-1
    # only difference: go to the end
    # may be 10-100 more words there (we used floor on the step)
    chunk = unicode(' ').join(words[(i*step):])
    textValence,textFvec = emotion(chunk,labMT,shift=True,happsList=labMTvector)
    stoppedVec = stopper(textFvec,labMTvector,labMTwordList,stopVal=2.0)
    timeseries[i] = emotionV(stoppedVec,labMTvector)

    timeseries = coursegrain(timeseries,points=21)

    g = open(outfile,"w")
    g.write("{0:.0f}".format(timeseries[0]))
    for i in xrange(1,numPoints):
        g.write(",")
        g.write("{0:.0f}".format(timeseries[i]))
    g.write("\n")
  
def processall(count):
    f = open("diskBooksData.txt","r")
    # skip the header
    f.readline()
    
    # skip count lines
    for i in xrange(count):
        f.readline()

    tmp = f.readline().rstrip().split('\t')
    f.close()

    print "reading:"
    print tmp

    # check that we got a language out
    if len(tmp) > 3:
        lang = tmp[3].lower()
    else:
        lang = "unknown"
      
    # we'll capture this one
    if lang == "en":
        lang = "english"

    print "language is "+lang

    # check if we have the language
    if lang in ["arabic","chinese","english","french","german","hindi","indonesian","korean","pashto","portuguese","russian","spanish","urdu"]:
      
        saveas = "timeseries/"+tmp[0]+".csv"
        count+=1
        labMT,labMTvector,labMTwordList = emotionFileReader(stopval=0.0,fileName='labMT2'+lang+'.txt',returnVector=True)
        # use the extracted books over on Jake's directory
        g = codecs.open("/users/j/r/jrwillia/scratch/data/gutenberg/books/"+tmp[0]+".txt","r","utf8")
        raw_text = g.read()
        g.close()
          
        words = [x.lower() for x in re.findall(r"[\w\@\#\'\&\]\*\-\/\[\=\;]+",raw_text,flags=re.UNICODE)]
          
        # avhapps = emotion(raw_text,labMT)
        print "there are "+str(len(words))+" words"
      
        if len(words) > 20000:
            chopper(words,labMT,labMTvector,labMTwordList,saveas)
            # append to the log
            h = open("analysis-log.csv","a")
            h.write("{0},{1},{2},{3}".format(tmp[0],tmp[1],tmp[2],lang))
            h.write("\n")
            h.close()

        # if len(words) > 50000:
    # if lang in ["arabic",...,"urdu"]:


if __name__ == "__main__":
    count = int(sys.argv[1])
    processall(count)



