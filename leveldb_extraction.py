'''
Created on Jan 9, 2016

@author: tim
'''
from leveldbX import LevelDBX
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import cPickle as pickle
import numpy as np
import gc
from util3 import Util
from sklearn.metrics.pairwise import linear_kernel
from sklearn.preprocessing import Normalizer
import wikipedia


u = Util()

extract_vocabulary = True
dump_titles = False
dump_documents = False
fit_tfidf = False
transform_tfidf = False
svd_transform = False
extract_train = False

predict = False

debug = True

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
    
    tfidf = TfidfVectorizer(max_features=20000,stop_words='english')
    
    t0 = time.time()
    X = []
    with open('/home/tim/wiki2/documents0.txt','r') as f:
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
    with open('/home/tim/wiki2/documents0.txt','r') as f:
        for i, line in enumerate(f):
            X.append(line)
    
    with open('/home/tim/wiki2/documents1.txt','r') as f:
        for i, line in enumerate(f):
            X.append(line)
      
    '''
    32GB not enough      
    with open('/home/tim/wiki2/documents2.txt','r') as f:
        for i, line in enumerate(f):
            X.append(line)
    '''     
    
    print "Read all files in {0}sec".format(time.time() - t0)
    t0 = time.time()    
    
    sparse = tfidf.transform(X)
    u.save_sparse_matrix('/home/tim/wiki2/tfidf.p', sparse)
    print "Transformed in {0}sec".format(time.time() - t0)#takes 30 minutes
    
    
if svd_transform:
    svd = TruncatedSVD(n_components= 200)
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



if extract_train:
    with open('training_set.tsv','r') as f:
        train = f.readlines()[1:]
        
        
    ids = []
    qs = []
    targets = []
    choices = []
    for q in train:
        
        line =  q.strip().split('\t')
        ids.append(line[0])
        qs.append(line[1])
        targets.append(line[2])
        choices.append(line[3:])
        
    choices = np.array(choices)
    
    data = {}
    data['ids'] = ids
    data['X'] = qs
    data['y'] = targets
    data['choices'] = [choices[:,0], choices[:,1], choices[:,2], choices[:,3]]
    
    pickle.dump(data, open('train_data.p','wb'),pickle.HIGHEST_PROTOCOL)   
    

if predict:
    print 'loading data..'
    data = pickle.load(open('train_data.p'))
    targets = data['y']
    
    titles = np.array(pickle.load(open('/home/tim/wiki2/titles.p')))
    
    tfidf = pickle.load(open('/home/tim/wiki2/model_tfidf.p'))
    #svd = pickle.load(open('svd.p'))
    #norm = pickle.load(open('norm.p'))
    
    #wikipedia = u.load_hdf5_matrix("/home/tim/data/allenai/svd_data.hdf5")
    wikipedia = u.load_sparse_matrix('/home/tim/wiki2/tfidf.p')
    
    X = tfidf.transform(data['X'])
    #X = svd.transform(X)
    #X = norm.transform(X)
    
    y1 = tfidf.transform(data['choices'][0])    
    #y1 = svd.transform(y1)
    #y1 = norm.transform(y1)
    
    y2 = tfidf.transform(data['choices'][1])    
    #y2 = svd.transform(y2)
    #y2 = norm.transform(y2)
    
    y3 = tfidf.transform(data['choices'][2])
    #y3 = svd.transform(y3)
    #y3 = norm.transform(y3)
    
    y4 = tfidf.transform(data['choices'][3])
    #y4 = svd.transform(y4)
    #y4 = norm.transform(y4)
    
    
    answers = ['A','B','C','D']
    correct = 0
    func = linear_kernel
    for i in range(X.shape[0]):
        q = X[i]
        a1 = y1[i]
        a2 = y2[i]
        a3 = y3[i]
        a4 = y4[i]
        
        dist = func(q, wikipedia).flatten()
        print dist.shape
        top10 = np.argsort(dist)[:-5:-1]
        
        print top10.shape
        selection = wikipedia[top10]
        
    
    
        print np.sum(func(selection, a1))
        print np.sum(func(selection, a2))
        print np.sum(func(selection, a3))
        print np.sum(func(selection, a4))
        idx = np.argmax([np.sum(func(selection, a1)),np.sum(func(selection, a2)),np.sum(func(selection, a3)),np.sum(func(selection, a4))])
        
        correct += answers[idx] == targets[i] 
        
        print targets[i]
        print data['X'][i]
        print type(titles)
        print titles[top10]
        print '------------------'
        print correct/(i+1.0)
    
    
        
        
    
