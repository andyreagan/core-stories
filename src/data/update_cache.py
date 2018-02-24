
# coding: utf-8

# In[1]:

from os import listdir, mkdir
from os.path import isfile, join, isdir
from json import loads
from re import findall,UNICODE
import sys
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

from tqdm import tqdm

from sys import argv
# In[4]:

filters = {"min_dl":int(argv[1]),
           "length": [20000,100000],
           "P": True,
           "n_points": 200,
           "salad": False,
          }
q = get_books(Book,filters)
version_str = get_version_str(filters)
stop_val = 1.0
        
books_to_refresh = []
for i in tqdm(range(len(q))):
    b = q[i]
    if not isfile(join("/Users/andyreagan/projects/2014/09-books/data/cache",str(b.pk)+".p.lz4")):
        books_to_refresh.append(b)
for i in tqdm(range(len(books_to_refresh))):
    b = books_to_refresh[i]
    b_data = Book_raw_data(b)
    b_data.chopper_sliding(my_LabMT,num_points=200,stop_val=stop_val,randomize=filters["salad"])
    save_book_raw_data(b_data)

