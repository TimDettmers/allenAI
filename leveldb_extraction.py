'''
Created on Jan 9, 2016

@author: tim
'''
from leveldbX import LevelDBX
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import cPickle as pickle
import numpy as np
import gc
from util3 import Util
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import Normalizer
import wikipedia
import nltk
from nltk.tokenize import wordpunct_tokenize
import gc

#nltk.download_gui()
    

extract_vocabulary = False
dump_titles = False
dump_documents = False
fit_tfidf = False
transform_tfidf = False
svd_transform = False
extract_data = False

predict = True

predict_train = True

debug = False



class SnowballTokenizer(object):
    def __init__(self):
        self.wnl = nltk.stem.SnowballStemmer("english")
    def __call__(self, doc):
        return [self.wnl.stem(t) for t in wordpunct_tokenize(doc)]  
    
def parse_data_set(path, isTrainset=True):
    with open(path,'r') as f:
        train = f.readlines()[1:]
        
        
    ids = []
    qs = []
    targets = []
    choices = []
    combined = []
    for q in train:
        
        line =  q.strip().split('\t')
        ids.append(line[0])
        qs.append(line[1].lower())
        if isTrainset:            
            targets.append(line[2])
            choices.append(line[3:])
        else:            
            choices.append(line[2:])
        
            
        
    choices = np.array(choices)
    
    data = {}
    data['ids'] = ids
    data['X'] = qs
    data['y'] = targets
    print choices
    data['choices'] = [choices[:,0], choices[:,1], choices[:,2], choices[:,3]]
    for i in range(len(ids)):
        combined.append([qs[i] + " " + choices[i,0], qs[i] + " " + choices[i,1], qs[i] + " " + choices[i,2], qs[i] + " " + choices[i,3]])
    data['combined'] = combined
    
    return data


def transform_data(data, withSVD=False):
    X = tfidf.transform(data['X'])
    
    y1 = tfidf.transform(data['choices'][0])    
    y2 = tfidf.transform(data['choices'][1])      
    y3 = tfidf.transform(data['choices'][2])    
    y4 = tfidf.transform(data['choices'][3])
    
    
    wikipedia = u.load_sparse_matrix('/home/tim/wiki2/tfidf.p')
    
    if withSVD:
        X = svd.transform(X)
        #X = norm.transform(X)
        y1 = svd.transform(y1)
        #y1 = norm.transform(y1)  
        y2 = svd.transform(y2)
        #y2 = norm.transform(y2)
        y3 = svd.transform(y3)
        #y3 = norm.transform(y3)
        y4 = svd.transform(y4)
        #y4 = norm.transform(y4)
        wikipedia = u.load_hdf5_matrix("/home/tim/data/allenai/svd_data.hdf5")
    
    return X, y1, y2, y3, y4, wikipedia

def transform_data2(data, withSVD=False):
    
    print np.array(data['X']).shape
    
    combined = np.array(data['combined'])
    print combined.shape
    
    y1 = tfidf.transform(combined[:,0]) 
    y2 = tfidf.transform(combined[:,1])      
    y3 = tfidf.transform(combined[:,2])
    y4 = tfidf.transform(combined[:,3])  
    
    print y1.shape
    
    wikipedia = u.load_sparse_matrix('/home/tim/wiki2/tfidf.p')
    
    if withSVD:
        y1 = svd.transform(y1)
        y2 = svd.transform(y2)
        y3 = svd.transform(y3)
        y4 = svd.transform(y4)
        wikipedia = u.load_hdf5_matrix("/home/tim/data/allenai/svd_data.hdf5")
    
    return y1, y2, y3, y4, wikipedia

u = Util()


documents = ["/home/tim/wiki2/documents0.txt","/home/tim/wiki2/documents1.txt","/home/tim/wiki2/documents2.txt"]
documents = ["/home/tim/data/allenai/wiki_filtered.txt"]

if debug:
    db = LevelDBX(path="/home/tim/wiki2")    
    data = db.get_table("raw_pages")    
    
    print data.get('Firefly')['raw']

if extract_vocabulary:    
    data = pickle.load(open('train_data.p'))
    
    vocab = {}
    
    X = data['X']
    
    
    for q in X:        
        for word in q.replace('?','').replace('.','').split(" "):
            if word not in vocab: vocab[word] = 0
            vocab[word] += 1
            
    print len(vocab.keys())
    
            
    


if dump_titles:
    titles = []
    db = LevelDBX(path="/home/tim/wiki2")    
    data = db.get_table("raw_pages")    
    documents = []
    
    t0 = time.time()
    for i, (title, page) in enumerate(data.scan()):
        if i % 100000 == 0: 
            print i 
        if isinstance(page, dict):
            titles.append(title)
        

    pickle.dump(titles, open('/home/tim/wiki2/titles.p','wb'), pickle.HIGHEST_PROTOCOL)



if dump_documents:

    db = LevelDBX(path="/home/tim/wiki2")    
    data = db.get_table("raw_pages")    
    documents = []
    
    t0 = time.time()
    for i, (title, page) in enumerate(data.scan()):
        if i % 100000 == 0: 
            print i 
        if isinstance(page, dict):
            documents.append(page['raw'])
        
        if i > 0 and i % (1000000) == 0:    
            print 'writing dump...'
            with open('/home/tim/wiki2/documents{0}.txt'.format(i/1000000),'wb') as f:
                for doc in documents:                
                    doc = re.sub('\n', '',doc)
                    f.write(doc.encode('utf-8') + "\n")            
            print 'freeing documents...'
            del documents[:]
            print len(documents)
            gc.collect()




if fit_tfidf:
    tfidf = TfidfVectorizer(max_features=50000, strip_accents='unicode',  
        analyzer='word',token_pattern=r'\w{3,}',sublinear_tf=1,
        ngram_range=(1, 1),#tokenizer = SnowballTokenizer(),
         stop_words = 'english',dtype=np.int32)
    
    
    t0 = time.time()
    X = []
    with open(documents[0],'r') as f:
        for i, line in enumerate(f):
            X.append(line)
    print "Read in {0}sec".format(time.time() - t0)
    
    print len(X)
    
    t0 = time.time()    
    sparse = tfidf.fit_transform(X)
    
    print "Transformed in {0}sec".format(time.time() - t0)
    
    
    u.save_sparse_matrix('/home/tim/wiki2/tfidf.p', sparse)
    
    pickle.dump(tfidf, open('/home/tim/wiki2/model_tfidf.p','wb'), pickle.HIGHEST_PROTOCOL)
    

if transform_tfidf:
    
    tfidf = pickle.load(open('/home/tim/wiki2/model_tfidf.p','r'))
    
    X = []
    t0 = time.time()
    for document in documents:
        with open(document,'r') as f:
            for i, line in enumerate(f):
                X.append(line)
    print "Read all files in {0}sec".format(time.time() - t0)
    t0 = time.time()    
    
    sparse = tfidf.transform(X)
    u.save_sparse_matrix('/home/tim/wiki2/tfidf.p', sparse)
    print "Transformed in {0}sec".format(time.time() - t0)#takes 30 minutes
    
    
if svd_transform:
    svd = TruncatedSVD(n_components= 500)
    t0 = time.time()
    X = u.load_sparse_matrix('/home/tim/wiki2/tfidf.p')
    print "Loaded sparse matrix in {0}sec".format(time.time() - t0)
    t0 = time.time()
    X = svd.fit_transform(X)
    print "SVD fitted in {0}sec".format(time.time() - t0)
    
    norm = Normalizer()
    X = norm.fit_transform(X)    
    
    u.save_hdf5_matrix("/home/tim/data/allenai/svd_data.hdf5", X)
    pickle.dump(svd, open('svd.p','wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(norm, open('norm.p','wb'), pickle.HIGHEST_PROTOCOL)



if extract_data:

    train_data = parse_data_set('training_set.tsv')
    pickle.dump(train_data, open('train_data.p','wb'),pickle.HIGHEST_PROTOCOL)  
    
    test_data = parse_data_set('validation_set.tsv', False)
    pickle.dump(test_data, open('test_data.p','wb'),pickle.HIGHEST_PROTOCOL)   
    

if predict:
    print 'loading data..'
    train = pickle.load(open('train_data.p'))
    test = pickle.load(open('test_data.p'))
    targets = train['y']
    
    #titles = np.array(pickle.load(open('/home/tim/wiki2/titles.p')))
    with open('/home/tim/data/allenai/wiki_filtered_title.txt') as f:
        titles = []
        for title in f:
            if len(title.strip()) == 0: continue
            titles.append(title)
    titles = np.array(titles)
    
    tfidf = pickle.load(open('/home/tim/wiki2/model_tfidf.p'))
    svd = pickle.load(open('svd.p'))
    norm = pickle.load(open('norm.p'))
 
 
    if predict_train:
        X, y1, y2, y3, y4, wikipedia = transform_data(train, False)
    else:
        X, y1, y2, y3, y4, wikipedia = transform_data(test, False)
        
    X.data = np.float32(X.data)
    wikipedia.data = np.float32(wikipedia.data)
    
    predictions = []
    answers = ['A','B','C','D']
    correct = 0
    func = linear_kernel    
    #func = lambda X, vec: np.sum((X-vec)**2,1)
    
    print 'calculating full kernel...'    
    gc.collect()
    dist1 = func(X[0:X.shape[0]/3], wikipedia)
    top1 = np.argsort(dist1,axis=1)[:,:-20:-1]
    del dist1
    gc.collect()
    dist2 = func(X[X.shape[0]/3:], wikipedia)
    top2 = np.argsort(dist2,axis=1)[:,:-20:-1]
    del dist2
    dist3 = func(X[X.shape[0]/3:], wikipedia)
    top3 = np.argsort(dist3,axis=1)[:,:-20:-1]
    del dist3
    gc.collect()
    top = np.vstack([top1, top2, top3])
    del top1
    del top2
    del top3
    gc.collect()
    
    
    
    #selection2 = wikipedia.todense()[top2]
    mats = []    
    for i in range(top.shape[1]):
        mats.append(wikipedia[top[:,i]])
    for i in range(X.shape[0]):
        q = X[i]
        a1 = y1[i]
        a2 = y2[i]
        a3 = y3[i]
        a4 = y4[i]
        
        
        s1=0
        s2=0
        s3=0
        s4=0
        for j in range(top.shape[1]):
            selection = mats[j][i]
            s1+= np.sum(func(a1, selection))
            s2+= np.sum(func(a2, selection))
            s3+= np.sum(func(a3, selection))
            s4+= np.sum(func(a4, selection))
        idx = np.argmax([s1,s2,s3,s4])
        
        if predict_train:
            correct += answers[idx] == targets[i]
            print targets[i]
        
        if predict_train:
            print correct/(i+1.0)
            print train['X'][i]
            sum_value = s1 + s2 + s3 + s4
            predictions.append([train['ids'][i], s1/sum_value, s2/sum_value, s3/sum_value, s4/sum_value])
        else:
            print test['ids'][i]
            print test['X'][i]
            predictions.append([test['ids'][i],answers[idx]])
        #print predictions[-1]
        print '------------------'        
       
    if predict_train:
        with open("/home/tim/data/allenai/results_probabilities.csv",'wb') as f:
            for id, s1,s2,s3,s4 in predictions:
                f.write("{0},{1},{2},{3},{4}\n".format(id, s1,s2,s3,s4))
    else:
        with open('/home/tim/data/allenai/result.csv','wb') as f:
       
            f.write('id,correctAnswer\n')
            for id, pred in predictions:
                f.write("{0},{1}\n".format(id, pred))
   
        
   
    
        
        
    
