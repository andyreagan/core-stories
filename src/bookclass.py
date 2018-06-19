from os import listdir
from os.path import isfile, join, isdir
from json import loads
from sys import path
path.append("/Users/andyreagan/tools/python")
from kitchentable.dogtoys import *
from numpy import dot,cumsum,floor,zeros,sum,array,random,ones
from labMTsimple.labMTsimple.storyLab import *
from labMTsimple.labMTsimple.speedy import LabMT
my_LabMT = LabMT()
from scipy.sparse import lil_matrix,issparse
# notes on sparse matrices:
# use the lil_matrix to build it out
# then convert to csr for storage/operations
import pickle
import lz4
import itertools
from tqdm import tqdm
import re
from copy import deepcopy
from spacy.en import English
nlp = English()

base_dir = "/Users/andyreagan/projects/2014/09-books/"
use_compression = True

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


def get_maintext_lines_gutenberg(raw_text):
    lines = raw_text.split("\n")
    start_book_i = 0
    end_book_i = len(lines)-1
    # pass 1, this format is easy and gets 78.9% of books
    start1="START OF THIS PROJECT GUTENBERG EBOOK"
    start2="START OF THE PROJECT GUTENBERG EBOOK"
    end1="END OF THIS PROJECT GUTENBERG EBOOK"
    end2="END OF THE PROJECT GUTENBERG EBOOK"
    end3="END OF PROJECT GUTENBERG"
    for j,line in enumerate(lines):
        if (start1 in line) or (start2 in line):
            # and "***" in line and start_book[i] == 0 and j<.25*len(lines):
            start_book_i = j
        end_in_line = end1 in line or end2 in line or end3 in line.upper()
        if end_in_line and (end_book_i == (len(lines)-1)):
            #  and "***" in line and j>.75*len(lines)
            end_book_i = j
    # pass 2, this will bring us to 99%
    if (start_book_i == 0) and (end_book_i == len(lines)-1):
        for j,line in enumerate(lines):
            if ("end" in line.lower() or "****" in line) and  "small print" in line.lower() and j<.5*len(lines):
                start_book_i = j
            if "end" in line.lower() and "project gutenberg" in line.lower() and j>.75*len(lines):
                end_book_i = j
        # pass three, caught them all (check)
        if end_book_i == len(lines)-1:
            for j,line in enumerate(lines):
                if "THE END" in line and j>.9*len(lines):
                    end_book_i = j
    return lines[(start_book_i+1):(end_book_i)]

def chunkify(lines):
    # put them back together...
    raw_text = "\n".join(lines)
    # remove extra whitespace
    raw_text_1 = re.sub("\n[\\s]+?\n","\n\n",raw_text)
    # remove singleton newlines
    raw_text_2 = re.sub(r"([^\n])\n([^\n])",r"\1 \2",raw_text_1)
    raw_text_3 = raw_text_2.rstrip().lstrip()

    # three levels of significance
    # single newlines were already discarded (insignificant)
    # double newlines are real line breaks
    # triple newlines (or more) separate content

    # split on those triples (or more)
    big_chunks = re.split("\n\n\n+",raw_text_3)

    # now break them into the paragraphs
    small_chunks = list(map(lambda x: re.split("\n\n",x),big_chunks))
    # combined_chunks = []
    # [combined_chunks.extend(el) for el in small_chunks]
    combined_chunks = []

    for i in range(len(small_chunks)):
        for j in range(len(small_chunks[i])):
            combined_chunks.append((i,j,small_chunks[i][j]))
    return combined_chunks

honorifics = ["Mr","Master","Miss","Ms","Mrs","Mx",
              "Sir","Madame","Dame","Lord","Lady","Esq","Adv",
              "Dr","Prof",
              "Rev","Fr","Pr","Br","Sr","Elder","Rabbo"]
abbreviations = ["J.R.R"] # ,"Ph.D","M.S"] # <- could end a sentence...
def replace_honorifics(s):
    for h in honorifics+abbreviations:
        s = re.sub(h+"\\.",h,s)
    return s

sentence_re = re.compile('[\\s\n]*(.+?[\\.!?]+["’]*(?=\\s+[A-Z"’‘]|$))',flags=re.DOTALL)
def find_sentences_with_RE(s):
    return sentence_re.findall(replace_honorifics(s))

def gen_text(model,seeds,n_words=20000):
    i = 2
    result = random.choice(seeds).split(" ")
    while i<n_words:
        # print(result)
        while " ".join(result[-2:]) not in model:
            # result[-1] += "."
            result.extend(random.choice(seeds).split(" "))
        result.append(random.choice(model[" ".join(result[-2:])][0],p=model[" ".join(result[-2:])][1]))
        i+=1
    return result

def train_markov_model(sentences):
    model = dict()
    starts = list()
    # let's iterate over sentences
    for sent in sentences:
        sent_tokens = [t for t in sent if (not t.is_punct or str(t).rstrip() == ",")]
        if len(sent_tokens) < 2:
            continue
        # print(sent_tokens)
        i = 0
        starts.append((str(sent_tokens[i]).rstrip()+ " " +str(sent_tokens[i+1]).rstrip()))
        for i in range(len(sent_tokens)-2):
            bigram = (str(sent_tokens[i]).rstrip()+ " " +str(sent_tokens[i+1]).rstrip())
            nextg = str(sent_tokens[i+2]).rstrip()
            if i == len(sent_tokens)-3:
                nextg += "."
            if bigram not in trigram_model:
                trigram_model[bigram] = {nextg: 1}
            else:
                if nextg in trigram_model[bigram]:
                    trigram_model[bigram][nextg] += 1
                else:
                    trigram_model[bigram][nextg] = 1
    return trigram_model,starts

def markov_text(chunks,n_words):
    sentences = []
    for c in chunks:
        sentences.extend(c[2].sents)
    trigram_model,starts = train_markov_model(sentences)
    trigram_model_flat = dict()
    for bg in trigram_model:
        words = [w for w in trigram_model[bg]]
        ps = [trigram_model[bg][w] for w in words]
        trigram_model_flat[bg] = [words,array(ps)/array(ps).sum()]
    markov_tokens = gen_text(trigram_model_flat,starts,n_words=n_words)
    markov_tokens_no_punct = [x.rstrip(" ").rstrip(".") for x in markov_tokens if not x.rstrip(" ") == ","]
    return markov_tokens_no_punct

class Book_raw_data(object):
    '''Book class to handle loading the calibre expanded epub format.
    Or just a text file.

    Now closely wrapping the database.

    Initialize with an instrance of the database like this:
    b = Book.objects.all()[0]
    b_data = Book_raw_data(b)

    Store all of the word lists, etc, in one place.'''

    def load_all_combined(self):
        # didn't do a perfect job creating the updated gutenberg
        # database...so still watch out for bad encodings
        filename = join("/Users/andyreagan/projects/2014/09-books/",self.this_Book.txt_file_path)
        if not isfile(filename):
            print(filename)
            raise("Hey man, why you no haz file?")
        try:
            # print("opening:",filename)
            f = open(filename,"r")
            rawtext = f.read()
            f.close()
        except:
            # print("failed, opening with iso8859:",filename)
            f = open(filename,"r",encoding="iso8859")
            rawtext = f.read()
            f.close()
            # print("writing as unicode:",filename)
            f = open(filename,"w")
            f.write(rawtext)
            f.close()
        # self.all_word_list = listify(rawtext)
        lines = get_maintext_lines_gutenberg(rawtext)
        # print("found",len(lines),"lines")
        chunked = chunkify(lines)
        # go get the sentences
        # combined_chunk_sentences = []
        # _ = [combined_chunk_sentences.extend([[x[0],x[1],j,y] for j,y in enumerate(find_sentences_with_RE(x[2]))]) for x in combined_chunks]
        # now, make the word list from the sentences
        # ...
        # alternatively, let's apply spacy's parser to the whole thing
        self.chunks_nlp = list(map(lambda x: [x[0],x[1],nlp(x[2]),x[2]],chunked))
        # all_token_list = []
        for chunk in self.chunks_nlp:
            for token in chunk[2]:
                if (not token.is_punct) and (len(str(token)) != 0):
                    self.all_word_list.append(str(token)) # get token.orth_
        # have to do this here, while the chunks object still kicks around
        # self.random_word_list = markov_text(chunks_nlp,len(all_token_list))
        # self.all_word_list = all_token_list

    def make_markov_text(self):
        self.random_word_list = markov_text(self.chunks_nlp,len(self.all_word_list))

    def make_salad_text(self):
        self.random_word_list = deepcopy(self.all_word_list)
        random.shuffle(self.random_word_list)

    def make_null_text(self,method):
        if method=="markov":
            self.make_markov_text()
        elif method=="salad":
            self.make_salad_text()
        else:
            print("method",method,"not implemented")

    def load_all_chapters(self):
        """Load data from text files in a folder.

        Will load just the .txt files"""
        folder = join("/Users/andyreagan/projects/2014/09-books/",self.this_Book.expanded_folder_path)

        # self.files = listdir(join("data/Kindle-combined-txt",str(isbn)))
        # no reason to get too complicated here
        self.txtfiles = [x for x in listdir(folder) if ".txt" in x]

        # print("a sample of the text files:")
        # print(self.txtfiles[:10])

        rawtext_by_chapter = []
        for fname in self.txtfiles:
            f = open(join(folder,fname),"r")
            rawtext_by_chapter.append(f.read())
            f.close()
        # word_lists_by_chapter = [listify(t) for t in rawtext_by_chapter]
        # apply spacy to each of them...
        word_lists_by_chapter = [[str(token) for token in nlp(t) if ((not token.is_punct) and (len(str(token)) != 0))] for t in rawtext_by_chapter]
        self.chapter_ends = cumsum(list(map(len,word_lists_by_chapter)))
        # add a 0 to the start, clip (to get the starts)
        # could just move the above array around too...
        self.chapter_beginnings = cumsum([0]+list(map(len,word_lists_by_chapter[:-1])))
        self.chapter_centers = (self.chapter_ends+self.chapter_beginnings)/2
#         print(list(map(len,self.word_lists_by_chapter)))
#         print(self.chapter_ends)
#         print(self.chapter_beginnings)
#         print(self.chapter_centers)
#         print(len(self.chapter_ends))
#         print(len(self.word_lists_by_chapter))
        self.all_word_list = list(itertools.chain(*word_lists_by_chapter))

    def coursegrain(t,points=21):
        # take a vector of scores
        # and break it down into a series of points
        extent = [min(t),max(t)]
        # print(extent)
        # create the bins
        nbins = float(points)
        nbinsize = (extent[1]-extent[0])/nbins
        # print(nbinsize)
        binsize = nbinsize + nbinsize*2/nbins
        # print(binsize)
        bins = [extent[0]+i*binsize-nbinsize*2/nbins for i in range(points)]
        # print(bins)
        # print(bins)
        # newvec = [binn(bins,v) for v in t]
        # print(newvec)

        # normalize starting point to 0
        tmpb = [binn(bins,v) for v in t]
        # return tmpb
        return [x-tmpb[0] for x in tmpb]

    def chop(self,my_senti_dict,min_size=1000,stop_val=0.0): #,save=False,outfile=""):
        """Take long piece of text and generate the sentiment time series.

        use save parameter to write timeseries to a file."""

        # print("splitting the book into chunks of minimum size {}".format(min_size))
        # lots o redundancy
        # self.all_words = " ".join(self.rawtext_by_chapter)
        # self.all_word_list = listify(self.all_words)
        # some redundancy
        # all_words = " ".join(self.rawtext_by_chapter)
        # self.all_word_list = listify(all_words)

        self.all_fvecs = []

        for i in range(int(floor(len(self.all_word_list)/min_size))):
            chunk = ""
            if i == int(floor(len(self.all_word_list)/min_size))-1:
                # take the rest
                # print('last chunk')
                # print('getting words ' + str(i*min_size) + ' through ' + str(len(self.all_word_list)-1))
                for j in range(i*min_size,len(self.all_word_list)-1):
                    chunk += self.all_word_list[j]+" "
            else:
                # print('getting words ' + str(i*min_size) + ' through ' + str((i+1)*min_size))
                for j in range(i*min_size,(i+1)*min_size):
                    chunk += self.all_word_list[j]+" "
                # print(chunk[0:10])

            chunk_words = listify(chunk)
            chunk_dict = dict()
            for word in chunk_words:
                if word in chunk_dict:
                    chunk_dict[word] += 1
                else:
                    chunk_dict[word] = 1
            text_fvec = my_senti_dict.wordVecify(chunk_dict)

            # print chunk
            # print 'the valence of {0} part {1} is {2}'.format(rawbook,i,textValence)

            self.all_fvecs.append(text_fvec)
            stoppedVec = stopper(text_fvec,my_senti_dict.scorelist,my_senti_dict.wordlist,stopVal=stop_val)
            self.timeseries.append(dot(my_senti_dict.scorelist,stoppedVec)/sum(stoppedVec))
            self.all_fvecs = csr_matrix(self.all_fvecs)
        return self.timeseries

    def chopper_sliding(self,my_senti_dict,min_size=10000,num_points=100,stop_val=0.0,use_cache=False,randomize=False,random_method="markov"):
        """Take long piece of text and generate the sentiment time series.
        We will now slide the window along, rather than make uniform pieces.

        If the pickle of the timeseries and the sentiment vectors both exist, it will load and return these.
        If only the sentiment vectors exist, it will use these to make the timeseries specified.
        If neither exists, it will save both.

        If fvec_preloaded, won't return the window centers, since it never touches the raw text file.
        We don't touch it, because that's a costly procedure, to re-tokenize everything.
        """

        # force not to save the randomized version...
        if randomize:
            use_cache = False

        # print("and printing those frequency vectors")
        if use_compression:
            cache_file = join(self.cache_dir,"{0}-all-fvecs-{1}-{2}.p.lz4".format(self.this_Book.pk,min_size,num_points))
        else:
            cache_file = join(self.cache_dir,"{0}-all-fvecs-{1}-{2}.p".format(self.this_Book.pk,min_size,num_points))
        if isfile(cache_file) and use_cache:
            # print("loading fvec from cache")
            f = open(cache_file,"rb")
            if use_compression:
                self.all_fvecs = pickle.loads(lz4.decompress(f.read()))
            else:
                self.all_fvecs = pickle.loads(f.read())
            f.close()
            if not issparse(self.all_fvecs):
                self.all_fvecs = csr_matrix(self.all_fvecs)
            # print(all_fvecs)
        else:
            if randomize:
                if (len(self.all_word_list) == 0) or (len(self.chunks_nlp) == 0):
                    if isdir(join("/Users/andyreagan/projects/2014/09-books/",self.this_Book.expanded_folder_path)):
                        print("found expanded directory, going to load all of the chapters from there")
                        self.load_all_chapters()
                    else:
                        self.load_all_combined()
                self.make_null_text(random_method)
                word_list = deepcopy(self.random_word_list)
            else:
                if len(self.all_word_list) == 0:
                    if isdir(join("/Users/andyreagan/projects/2014/09-books/",self.this_Book.expanded_folder_path)):
                        print("found expanded directory, going to load all of the chapters from there")
                        self.load_all_chapters()
                    else:
                        self.load_all_combined()
                word_list = deepcopy(self.all_word_list)
            # first, just build the matrix
            step = int(floor((len(word_list)-min_size)/(num_points-1)))
            # print("there are "+str(len(self.all_word_list))+" words in the book")
            # print("step size "+str(step))
            self.centers = [i*step+(min_size)/2 for i in range(num_points)]

            def build_matrix(num_points,word_list,step,min_size,my_senti_dict):
                '''Build the matrix of overlapping word vectors.'''
                # disregard the existing matrix, it might not be the right size
                all_fvecs = lil_matrix((num_points,len(my_senti_dict.scorelist)),dtype="i")
                for i in range(num_points-1):
                    all_fvecs[i,:] = my_senti_dict.wordVecify(dictify(word_list[(i*step):(min_size+i*step)]))
                i = num_points-1
                all_fvecs[i,:] = my_senti_dict.wordVecify(dictify(word_list[(i*step):]))
                return all_fvecs

            self.all_fvecs = build_matrix(num_points,word_list,step,min_size,my_senti_dict).tocsr()
            # since the cache file didn't exist...
            # if use_cache and (not isfile(cache_file)):
            if use_cache:
                # print("saving fvec cache")
                f = open(cache_file,"wb")
                if use_compression:
                    f.write(lz4.compress(pickle.dumps(self.all_fvecs,pickle.HIGHEST_PROTOCOL)))
                else:
                    f.write(pickle.dumps(self.all_fvecs,pickle.HIGHEST_PROTOCOL))
                f.close()

        # all_fvecs_stopped = stopper_mat(self.all_fvecs,my_senti_dict.scorelist,my_senti_dict.wordlist,stopVal=stop_val)
        # initialize self.timeseries, only thing we're really after
        self.timeseries = [0 for i in range(num_points)]

        # # randomize it!!
        # # print(self.all_word_list[:10])
        # if randomize:
        #     use_cache = False
        #     random.shuffle(self.all_word_list)
        #     # print(self.all_word_list[:10])
        # if randomize:
       #     self.all_word_list = self.random_word_list

        # could probably vectorize this...
        for i in range(num_points):
            text_fvec = self.all_fvecs[i,:].toarray().squeeze()
            stoppedVec = stopper(text_fvec,my_senti_dict.scorelist,my_senti_dict.wordlist,stopVal=stop_val)
            self.timeseries[i] = dot(my_senti_dict.scorelist,stoppedVec)/sum(stoppedVec)

        return self.timeseries

    def __init__(self,dbref):
        self.all_word_list = []
        self.random_word_list = []
        self.cache_dir = "/Users/andyreagan/projects/2014/09-books/data/cache"
        self.chunks_nlp = []
        # remember it's better to add by row
        # default to this size...
        self.all_fvecs = lil_matrix((200,10222),dtype="i")
        self.timeseries = None
        self.centers = None
        self.this_Book = dbref

        # if self.Book.expanded_folder_path:
        #     self.load_all_chapters()
        # elif self.Book.txt_file_path:
        #     self.load_all_combined()

        # print(str(self))

        # will need to handle the metadata more generally...
        # f = open(join("data/Kindle-combined-txt",str(isbn),"meta.json"),"r")
        # self.metadata = loads(f.read())
        # f.close()
        # # print("this is the metadata:")
        # # print(self.metadata)

    def __str__(self):
        if len(self.this_Book.title) > 0:
            return "Book_raw_data "+self.this_Book.title
        else:
            return "Book_raw_data (no title)"

    # this just doesn't seem to work? should.
    def save(self):
        # if there is already a path to this object, use it
        # else, go ahead and create it
        if not self.this_Book.pickle_object:
            save_path = join(self.cache_dir,"{0}.p".format(self.this_Book.pk))
            self.this_Book.pickle_object = save_path
            self.this_Book.save()
        pickle.dump(self,open(self.this_Book.pickle_object,"wb"),pickle.HIGHEST_PROTOCOL)

def save_book_raw_data(book_raw_data_obj):

    if use_compression:
        f = open(join("/Users/andyreagan/projects/2014/09-books/data/cache",str(book_raw_data_obj.this_Book.pk)+".p.lz4"),"wb")
        f.write(lz4.compress(pickle.dumps(book_raw_data_obj,pickle.HIGHEST_PROTOCOL)))
        f.close()
    else:
        f = open(join("/Users/andyreagan/projects/2014/09-books/data/cache",str(book_raw_data_obj.this_Book.pk)+".p"),"wb")
        f.write(pickle.dumps(book_raw_data_obj,pickle.HIGHEST_PROTOCOL))
        f.close()

def cache_check(b):
    return False
    if use_compression:
        return isfile(join("/Users/andyreagan/projects/2014/09-books/data/cache",str(b.pk)+".p.lz4"))
    else:
        return isfile(join("/Users/andyreagan/projects/2014/09-books/data/cache",str(b.pk)+".p"))

def load_book_raw_data(b):
    if use_compression:
        f = open(join("/Users/andyreagan/projects/2014/09-books/data/cache",str(b.pk)+".p.lz4"),"rb")
        return pickle.loads(lz4.decompress(f.read()))
    # f didn't get closed...
    else:
        return pickle.load(open(join("/Users/andyreagan/projects/2014/09-books/data/cache",str(b.pk)+".p"),"rb"))

def get_books(model,filters):
    q = model.objects.filter(exclude=False,
                                       length__gt=filters["length"][0],
                                       length__lte=filters["length"][1],
                                       downloads__gte=filters["min_dl"],
                                       lang_code_id=0,
                                       locc_with_P=filters["P"])
    return q

def get_version_str(filters):
    # version = "009"
    # version = "P-20K-100K-500-7pt"
    P_str=""
    if filters["P"]:
        P_str = "P-"
    salad_str=""
    if filters["salad"]:
        salad_str="-salad"
    version = "{0}{1:.0f}K-{2:.0f}K-{3}dl-{4}pt{5}".format(P_str,
                                                           filters["length"][0]/1000.0,
                                                           filters["length"][1]/1000.0,
                                                           filters["min_dl"],
                                                           filters["n_points"],
                                                           salad_str)
    return version

def get_data(q,version,filters,use_cache=True):
    if isfile("/Users/andyreagan/projects/2014/09-books/data/gutenberg/timeseries-matrix-cache-{}.p".format(version)) and use_cache:
        big_matrix = pickle.load(open("/Users/andyreagan/projects/2014/09-books/data/gutenberg/timeseries-matrix-cache-{}.p".format(version),"rb"))
        if big_matrix.shape[0] == len(q):
            return big_matrix
    # load all of the timeseries into a matrix
    big_matrix = ones([len(q),200])
    # big_matrix_mean0 = np.ones(big_matrix.shape)
    stop_val = 1.0
    for i in tqdm(range(len(q))):
        b = q[i]
        # don't load book data with the salad version
        if cache_check(b) and (use_cache and not filters["salad"]):
            b_data = load_book_raw_data(b)
        else:
            b_data = Book_raw_data(b)
            b_data.chopper_sliding(my_LabMT,num_points=200,stop_val=stop_val,randomize=filters["salad"],use_cache=use_cache)
            # careful not to save individual book data
            if (use_cache and not filters["salad"]):
                save_book_raw_data(b_data)
        big_matrix[i,:] = b_data.timeseries
        # b_data.save()
    # print(big_matrix.shape)
    if use_cache:
        pickle.dump(big_matrix,open("/Users/andyreagan/projects/2014/09-books/data/gutenberg/timeseries-matrix-cache-{}.p".format(version),"wb"),pickle.HIGHEST_PROTOCOL)
    return big_matrix
