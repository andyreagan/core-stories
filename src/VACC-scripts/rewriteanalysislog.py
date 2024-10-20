# rewriteanalysislog.py
#
# USAGE
#
# python rewriteanalysislog.py

import codecs # handle utf8
import re
import sys
import os

def processlog():
    # look it up in the analysis log
    # f = open("analysis-log.csv","r")
    # g = open("analysislog.txt","w")
    f = open("timeserieslist.txt","r")
    g = open("analysislogactually.txt","w")

    for line in f:
        # booknum = line.rstrip().split(',')[0]
        booknum = line.rstrip('.csv\n')
        h = open("diskBooksData.txt","r")
        # skip the header
        h.readline()
        found = False
        while not found:
            line = h.readline()
            if line.split('\t')[0] == booknum:
                found = True
        h.close()
        g.write(line)
    
    f.close()
    g.close()
    
if __name__ == "__main__":
    processlog()








