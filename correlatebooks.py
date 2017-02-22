# /usr/share/nginx/wiki/mysite/mysite/static/hedonometer/data/bookdata/correlatebooks.py
#
# correlate the books
import codecs
from scipy.signal import correlate
from scipy.stats import pearsonr
from numpy import array

if __name__ == "__main__":
    # read in the list of books
    f = codecs.open("analysis-log.csv","r","utf8")
    booklist = [line.split(",") for line in f]
    f.close()

    print len(booklist)
    print booklist[0]
    print booklist[-1]

    # build a matrix of correlations
    # we'll fill in the upper triangle
    # default them all to 0
    numtocorrelate = 5 # len(booklist)
    correlations = [[0 for i in xrange(numtocorrelate)] for j in xrange(numtocorrelate)]
    
    for i in xrange(numtocorrelate):
        for j in xrange(i+1,numtocorrelate):
            f = open("timeseries/"+booklist[i][0]+".csv",'r')
            t1 = array(map(float,f.readline().split(",")))
            f.close()
            print len(t1)
            # print t1
            f = open("timeseries/"+booklist[j][0]+".csv",'r')
            t2 = array(map(float,f.readline().split(",")))
            f.close()
            print len(t2)
            # print t2
            # correlations[i][j] = correlate(t1,t2)
            correlations[i][j] = pearsonr(t1,t2)[0]

    print correlations
