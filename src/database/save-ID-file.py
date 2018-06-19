from os import listdir
from os.path import isfile, join
import sys
import json
from re import findall,UNICODE

import os
sys.path.append('/Users/andyreagan/projects/2014/09-books/database')
os.environ.setdefault('DJANGO_SETTINGS_MODULE','gutenbergdb.settings')
import django
django.setup()

from library.models import *

q = Book.objects.filter(exclude=False,length__gt=10000,length__lte=200000,downloads__gte=150,numUniqWords__gt=1000,numUniqWords__lt=18000,lang_code_id=0)

all_ids = [str(b.gutenberg_id) for b in q]
f = open("IDs.txt","w")
f.write("\n".join(all_ids))
f.close()
