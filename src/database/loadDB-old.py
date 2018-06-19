import sys, os
sys.path.append('/Users/andyreagan/work/2014/2014-09books/gutenbergdb')
os.environ.setdefault('DJANGO_SETTINGS_MODULE','gutenbergdb.settings')
import django
django.setup()

from library.models import Book

def load():
    f = open('diskBooksData.txt','r')
    f.readline()
    for line in f:
        fields = line.rstrip().split('\t')
        b = Book(filename=fields[0],
                 title=fields[1],
                 author=fields[2],
                 language=fields[3],
                 happs=0.0,
                 length=0,
                 ignorewords='',
                 wiki='http://example.com',
        )
        b.save()

def stat():
    allbooks = Book.objects.all() 
    print(len(allbooks))

if __name__ == '__main__':
    # load()
    stat()
