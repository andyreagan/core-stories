from os.path import isfile, join, isdir
import sys
from sys import argv
sys.path.append("/Users/andyreagan/tools/python")
from kitchentable.dogtoys import *
from labMTsimple.labMTsimple.speedy import LabMT
my_LabMT = LabMT()
from labMTsimple.labMTsimple.storyLab import *
import numpy as np
import pickle

import os
sys.path.append('/Users/andyreagan/projects/2014/09-books/database')
os.environ.setdefault('DJANGO_SETTINGS_MODULE','gutenbergdb.settings')
import django
django.setup()

from library.models import *
from bookclass import *

from tqdm import trange

from joblib import Parallel, delayed
import multiprocessing

def refresh_book(b):
    b_data = Book_raw_data(b)
    b_data.load_all_combined()
    # delete the cache_file for the fvec
    cache_file = join(b_data.cache_dir,"{0}-all-fvecs-{1}-{2}.p.lz4".format(b.pk,10000,200))
    if isfile(cache_file):
        os.remove(cache_file)
    b_data.chopper_sliding(my_LabMT,num_points=200,stop_val=1.0,randomize=False,use_cache=True)
    save_book_raw_data(b_data)
    b.length = len(b_data.all_word_list)
    b.numUniqWords = len(set(b_data.all_word_list))
    b.save()
    print(b.pk)
    
def refresh_data(q,filters):
    stop_val = 1.0
    for i in trange(len(q)):
        b = q[i]
        # don't load book data
        b_data = Book_raw_data(b)
        b_data.load_all_combined()
        # delete the cache_file for the fvec
        cache_file = join(b_data.cache_dir,"{0}-all-fvecs-{1}-{2}.p.lz4".format(b.pk,10000,filters["n_points"]))
        if isfile(cache_file):
            os.remove(cache_file)
        b_data.chopper_sliding(my_LabMT,num_points=filters["n_points"],stop_val=stop_val,randomize=False,use_cache=True)
        save_book_raw_data(b_data)
        b.length = len(b_data.all_word_list)
        b.numUniqWords = len(set(b_data.all_word_list))
        b.save()

if __name__ == "__main__":
    filters = {"min_dl":int(argv[1]),
               "length": [20000,100000],
               "P": True,
               "n_points": 200,
               "salad": False,
          }
    q = get_books(Book,filters)
    # refresh_data(q,filters)
    results = Parallel(n_jobs=4,verbose=40)(delayed(refresh_book)(b) for b in q)
    
