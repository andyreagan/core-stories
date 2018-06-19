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

# first, go extract all of the authors, with the PK
all_authors = Author.objects.all()
for a in all_authors[:10]:
    print((a.pk, a.fullname, a.note, a.gutenberg_id))
all_author_json = [{"pk": a.pk, "fullname": a.fullname, "note": a.note, "gutenberg_id": a.gutenberg_id} for a in all_authors]
f = open("all_author_info.json","w")
f.write(json.dumps(all_author_json,indent=4))
f.close()

# first, go extract all of the authors, with the PK
all_books = Book.objects.all()
for b in all_books[:10]:
    print((b.pk, b.title,))

all_book_json = [{"title": b.title,
                  "authors": [a.pk for a in b.authors.all()],
                  "language": b.language,
                  "lang_code_id": b.lang_code_id,
                  "downloads": b.downloads,
                  "gutenberg_id": b.gutenberg_id,
                  "mobi_file_path": b.mobi_file_path,
                  "epub_file_path": b.epub_file_path,
                  "txt_file_path": b.txt_file_path,
                  "expanded_folder_path": b.expanded_folder_path,
                  "length": b.length,
                  "numUniqWords": b.numUniqWords,
                  "ignorewords": b.ignorewords,
                  "exclude": b.exclude,
                  "excludeReason": b.excludeReason,}
                 for b in all_books]

f = open("all_book_info.json","w")
f.write(json.dumps(all_book_json,indent=4))
f.close()

