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


extract_vocabulary = False
dump_titles = False
dump_documents = False
fit_tfidf = False
transform_tfidf = False
svd_transform = False
extract_data = True

predict = True

debug = False


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
        qs.append(line[1])
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
    
    tfidf = TfidfVectorizer(max_features=20000,stop_words='english')
    
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
    svd = TruncatedSVD(n_components= 250)
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
 
 
    #X, y1, y2, y3, y4, wikipedia = transform_data(test, True)
    y1, y2, y3, y4, wikipedia = transform_data2(train, True)
    
    predictions = []
    answers = ['A','B','C','D']
    correct = 0
    func = linear_kernel    
    #func = lambda X, vec: np.sum((X-vec)**2,1)
    '''
    for i in range(X.shape[0]):
        q = X[i]
        a1 = y1[i]
        a2 = y2[i]
        a3 = y3[i]
        a4 = y4[i]
        
        dist = func(q, wikipedia).flatten()
        top = np.argsort(dist)[:-5:-1]        
        selection = wikipedia[top]
        weight = dist[top]/np.sum(dist[top])
        weight = np.exp(-np.arange(top.shape[0])/2.)
        weight = np.ones_like(weight)
        
        
    
        print np.sum(func(a1, selection)*weight)
        print np.sum(func(a2, selection)*weight)
        print np.sum(func(a3, selection)*weight)
        print np.sum(func(a4, selection)*weight)
        idx = np.argmax([np.sum(func(selection, a1)*weight),np.sum(func(selection, a2)*weight),np.sum(func(selection, a3)*weight),np.sum(func(selection, a4)*weight)])
        
        #correct += answers[idx] == targets[i]
        #print targets[i]
        
        print test['X'][i]
        print type(titles)
        print titles[top]
        print correct/(i+1.0)
        predictions.append([test['ids'][i],answers[idx]])
        #print predictions[-1]
        print '------------------'        
    '''
    for i in range(y1.shape[0]):        
        a1 = y1[i]
        a2 = y2[i]
        a3 = y3[i]
        a4 = y4[i]
        
        dist1 = func(a1, wikipedia).flatten()
        dist2 = func(a2, wikipedia).flatten()
        dist3 = func(a3, wikipedia).flatten()
        dist4 = func(a4, wikipedia).flatten()
        top1 = np.argsort(dist1)[:-2:-1]  
        top2 = np.argsort(dist2)[:-2:-1]  
        top3 = np.argsort(dist3)[:-2:-1]  
        top4 = np.argsort(dist4)[:-2:-1]
                
        print dist1[top1].sum()
        print dist2[top2].sum()
        print dist3[top3].sum()
        print dist4[top4].sum()
        idx = np.argmax([dist1[top1].sum(), dist2[top2].sum(), dist3[top3].sum(), dist4[top4].sum()])
        
        correct += answers[idx] == targets[i]
        print targets[i]
        
        print test['X'][i]
        print correct/(i+1.0)
        predictions.append([test['ids'][i],answers[idx]])
        #print predictions[-1]
        print '------------------'           
    with open('/home/tim/data/allenai/result.csv','wb') as f:
        f.write('id,correctAnswer\n')
        for id, pred in predictions:
            f.write("{0},{1}\n".format(id, pred))
        
   
    
        
        
    
