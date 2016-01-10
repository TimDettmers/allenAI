'''
Created on Jan 9, 2016

@author: tim
'''
from os import listdir
from os.path import isfile, join
import re
import operator
import numpy as np
import cPickle as pickle
import time

import sys
sys.path.insert(0, '/home/tim/git/clusterNet2/py_source/')
import cluster_net as gpu
from sklearn.feature_extraction.text import CountVectorizer




mypath = '/home/tim/data/wikidump/AA'

read_glove = False
read_vocab = True
expand_vocab = False

if read_glove:
    dataset = []
    keys = []
    t0 = time.time()
    with open('/home/tim/data/allenai/glove.6B.100d.txt','r') as f:
        for i, line in enumerate(f):
            if i % 100000 == 0: print i
            data = line.split(' ')
            tag = data[0]
            vector = np.array(data[1:],dtype=np.float32)
            dataset.append(vector)
            keys.append(tag)
            
           
            
            
    print time.time()-t0
            
           
    for key in keys[:3000]:
        print key
            
            
        
    
    X =  np.array(dataset, dtype=np.float32)
    #keys = keys[3000:]
    
  
    #space = gpu.VectorSpace(X, keys)
    #print space.find_nearest("vector and scalar quantities")
    #print space.find_nearest("einstein")
    #print space.find_nearest("einstein")
    
    #results = gpu.get_closest_index(np.array(dataset, dtype=np.float32),50)
    
    #pickle.dump(results, open('/home/tim/data/allenai/top_words.p','wb'),pickle.HIGHEST_PROTOCOL)
    
    
if read_vocab:
    vocab = {}
    with open('/home/tim/data/allenai/vocab.txt') as f:
        for word in f:
            words = word.strip().lower().replace("'s","").replace(',','').split(" ")
            for w in words:
                if w not in vocab: vocab[w] = 0
                vocab[w] += 1 
                
    vocab.pop("of")
    vocab.pop("and")
    vocab.pop("the")
    vocab.pop("in")
    vocab.pop("")     

if expand_vocab:
    space = gpu.VectorSpace(X, keys, stopwords=keys[3000:])
    
    expaned_vocab = {}
    '''
    with open('/home/tim/data/allenai/vocab.txt') as f:
        for phrase in f:
            if len(phrase.strip()) == 0: continue
            idx, words = space.find_nearest(phrase.replace("'s",''))
            print phrase, words
            for word in words:
                if word not in expaned_vocab: expaned_vocab[word] = 0
                expaned_vocab[word] += 1
    
    '''  
    print len(vocab.keys())
    for i, vocab_word in enumerate(vocab):
        if i % 100 == 0: print i
        idx, words = space.find_nearest(vocab_word.replace("'s",''))
        print vocab_word, words
        if words == None: continue
        for word in words:            
            if word not in expaned_vocab: expaned_vocab[word] = 0
            expaned_vocab[word] += 1  

    print len(expaned_vocab.keys())
    
    sorted_x = sorted(expaned_vocab.items(), key=operator.itemgetter(1), reverse=True)
    
    for word in sorted_x:
        print word
     
    pickle.dump(expaned_vocab, open('/home/tim/data/allenai/expanded_vocab.p','wb'))


#expaned_vocab = pickle.load(open('/home/tim/data/allenai/expanded_vocab.p','r'))
expaned_vocab = vocab
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

start = False

total = 0
included = 0
cntvec = CountVectorizer(stop_words="english",vocabulary=expaned_vocab.keys(), dtype=np.int32)

#space = gpu.VectorSpace(X, keys)
with open('/home/tim/data/allenai/wiki_filtered_title.txt','wb') as t:
    with open('/home/tim/data/allenai/wiki_filtered.txt','wb') as g:
        for path in onlyfiles:    
            with open(path) as f:
                page = []
                for line in f:   
                      
                    if start:
                        page.append(line)   
                            
                    if "<doc" in line[0:6]:
                        start = True
                        save = False
                        
                        
                    if "</doc" in line[0:6]:
                        if len(page) < 3: continue
                        total +=1
                        start = False
                        title = page[0]
                        #terms = title.strip().lower().split(' ')
                        matches = 0
                        
                        del page[0:2]
                        del page[-1]
                        '''
                        words = space.find_nearest(title)[1]
                        
                        if words == None: continue
                        count = 0                    
                        for word in words:
                            if word in expaned_vocab: count +=1
                            
                        print count, title
                        if count > 40: print words
                        '''
                        
                        #print doc
                        sum_value = np.sum(cntvec.transform(page).data.sum())
                        
                        if sum_value > 50 and "list" not in title.lower():
                            save=True
                        
                        '''
                        for term in terms:
                            if term in expaned_vocab:
                                matches +=1
                        if matches > 2:
                            save = True
                        '''     
                        
                        
                        if save:    
                            
                            included +=1 
                            doc = "".join(page)
                            doc = re.sub('\n', '',doc)
                            g.write(doc + "\n")
                            t.write(title)
                            
                        
                        
                        del page[:]
                        page = []
                    
                    if total > 0 and total % 10000 == 0:
                        total +=1
                        print included/float(total)
                        print total, included
           
