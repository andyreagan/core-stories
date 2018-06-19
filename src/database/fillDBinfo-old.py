import sys, os
sys.path.append('/Users/andyreagan/work/2014/2014-09books/gutenbergdb')
os.environ.setdefault('DJANGO_SETTINGS_MODULE','gutenbergdb.settings')
import django
django.setup()

from library.models import Book

from os.path import isfile
import re

sys.path.append("/Users/andyreagan/work/2014/03-labMTsimple/")
# for the VACC, doesn't hurt to have both
sys.path.append("/users/a/r/areagan/work/2014/03-labMTsimple/")
from labMTsimple.speedy import *
from labMTsimple.storyLab import *
my_labMT = LabMT(stopVal=2.0)

sys.path.append("/Users/andyreagan/work/2015/08-kitchentabletools/")
from dog.toys import *

import gzip

def checkfiles():
    for book in Book.objects.all():
        if isfile('../gutenberg/{0}.txt.gz'.format(book.filename)):
            pass
        else:
            book.exclude = True
            book.excludeReason = 'Couldnt find file'
            book.save()
    
    print(len(Book.objects.filter(exclude=True)))

def fillhappsinfo():
    i = 0
    # for book in Book.objects.all():
    for book in Book.objects.all()[:100]:
        f = gzip.open('../data/gutenberg/{0}.txt.gz'.format(book.filename),'r')
        words = listify(f.read())
        f.close()
        wordcounts = dictify(words)
        print(repr(u'{0} words in {1}'.format(len(words),book.title)))
        book.length = len(words)
        print(repr(u'{0} unique words in {1}'.format(len(wordcounts),book.title)))
        book.numUniqWords = len(wordcounts)
        
        # # score the whole thing
        # # I have the rest from the timeseries
        # if book.language.lower() == 'english':
        #     score = my_labMT.scoreTrie(wordcounts)
        #     print('{0} happs in {1}'.format(score,book.title))
        #     book.happs = score
        # else:
        #     book.happs = 0.0
        

        book.save()
        # some measure of progress on this
        i+=1
        print(i)
        

if __name__ == '__main__':
    # checkfiles()
    fillhappsinfo()
