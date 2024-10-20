# chop.py
#
# chop up a text into a bunch of frequency vectors of length
# 
# USAGE
#
# python chop.py data/count-of-monte-cristo.txt output french

import codecs # handle utf8
import re
from labMTsimple.storyLab import *
import sys
import os
sys.path.append('/usr/share/nginx/wiki/mysite/mysite')
sys.path.append('/usr/share/nginx/wiki/mysite')
os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'
from django.conf import settings

from hedonometer.models import Book

def chopper(words,labMT,labMTvector,outfile,minSize=1000):
  # print "now splitting the text into chunks of size 1000"
  # print "and printing those frequency vectors"
  allFvec = []
  from numpy import floor
  for i in xrange(int(floor(len(words)/minSize))):
    chunk = unicode('')
    if i == int(floor(len(words)/minSize))-1:
      # take the rest
      # print 'last chunk'
      # print 'getting words ' + str(i*minSize) + ' through ' + str(len(words)-1)
      for j in xrange(i*minSize,len(words)-1):
        chunk += words[j]+unicode(' ')
    else:
      # print 'getting words ' + str(i*minSize) + ' through ' + str((i+1)*minSize)
      for j in xrange(i*minSize,(i+1)*minSize):
        chunk += words[j]+unicode(' ')
        # print chunk[0:10]
    textValence,textFvec = emotion(chunk,labMT,shift=True,happsList=labMTvector)
      # print chunk
    # print 'the valence of {0} part {1} is {2}'.format(rawbook,i,textValence)
        
    allFvec.append(textFvec)


  f = open(outfile,"w")
  if len(allFvec) > 0:
    print "writing out the file to {0}".format(outfile) 
    f.write('{0:.0f}'.format(allFvec[0][0]))
    for k in xrange(1,len(allFvec)):
      f.write(',{0:.0f}'.format(allFvec[k][0]))
    for i in xrange(1,len(allFvec[0])):
      f.write("\n")
      f.write('{0:.0f}'.format(allFvec[0][i]))
      for k in xrange(1,len(allFvec)):
        f.write(',{0:.0f}'.format(allFvec[k][i]))
    f.close()
  else:
    print "\""*40
    print "could not write to {0}".format(outfile)
    print "\""*40
  # print "done!"
  
if __name__ == "__main__":
  # rawbook,saveas,lang = sys.argv[1:]
  # f = open("gutenberg/diskBooksData.txt","r")
  f = open("testbooks.txt","r")
  g = open("gutenberg/nolang.txt","w")
  count = 0
  # cont = False
  cont = True
  for line in f:
    tmp = line.rstrip().split("\t")
    lang = tmp[3].lower()
    # stopped here last time
    if tmp[0] == "0":
      cont = True
    # catch if its just the code
    if cont:
      if lang == "en":
        lang = "english"
      # check if we have the language
      if lang in ["arabic","chinese","english","french","german","hindi","indonesian","korean","pashto","portuguese","russian","spanish","urdu"]:
        saveas = tmp[0]+".csv"
        count+=1
        labMT,labMTvector,labMTwordList = emotionFileReader(stopval=0.0,fileName='labMT2'+lang+'.txt',returnVector=True)
        f = codecs.open("gutenberg/books/"+tmp[0]+".txt","r","utf8")
        raw_text = f.read()
        f.close()
  
        words = [x.lower() for x in re.findall(r"[\w\@\#\'\&\]\*\-\/\[\=\;]+",raw_text,flags=re.UNICODE)]
  
        avhapps = emotion(raw_text,labMT)
  
        chopper(words,labMT,labMTvector,saveas)
  
        b = Book(filename=tmp[0],title=tmp[1],author=tmp[2],language=lang,happs=avhapps,length=len(words))
        b.save()
      else:
        g.write(line)
      

  print count
  f.close()
  g.close()



