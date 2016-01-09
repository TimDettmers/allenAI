'''
Created on Jan 9, 2016

@author: tim
'''

#!/usr/bin/env python 
import sys, os, lucene 
from java.io import File 
from org.apache.lucene.analysis.standard import StandardAnalyzer 
from org.apache.lucene.index import DirectoryReader 
from org.apache.lucene.index import Term, IndexReader
from org.apache.lucene.queryparser.classic import QueryParser 
from org.apache.lucene.store import SimpleFSDirectory 
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search import Query, TermQuery 
from org.apache.lucene.util import Version 
""" 
This script is loosely based on the Lucene (java implementation) demo class  
org.apache.lucene.demo.SearchFiles.  It will prompt for a search query, then it 
will search the Lucene index in the current directory called 'index' for the 
search query entered against the 'contents' field.  It will then display the 
'path' and 'name' fields for each of the hits it finds in the index.  Note that 
search.close() is currently commented out because it causes a stack overflow in 
some cases. 
""" 


INDEX_DIR = "/home/tim/wiki2/lucene/index/"

def run(searcher, analyzer): 
    while True: 
        print 
        print "Hit enter with no input to quit." 
        command = raw_input("Query:") 
        if command == "": 
            return 
        print 
        print "Searching for:", command 
        """ 
        query = QueryParser(Version.LUCENE_CURRENT, "contents", 
                            analyzer).parse(command) 
        """ 
        query = TermQuery(Term("page", command)) 
        hits = searcher.search(query,10000) 
        print hits
        print "%s total matching documents." % hits.totalHits 
        print "Max score:",hits.getMaxScore() 
        for hit in hits.scoreDocs: 
            doc = searcher.doc(hit.doc)
            print doc.getField("page").stringValue() 
            #print 'URI:',doc.getField("path").stringValue() 
            #print 'File:',doc.getField('name').stringValue() 
            #print 'Digest:',doc.getField('contents').stringValue() 
            print 'Health:',hit.score
if __name__ == '__main__':
    lucene.initVM(vmargs=['-Djava.awt.headless=true'])
    print 'lucene', lucene.VERSION
    reader = IndexReader.open(SimpleFSDirectory(File(INDEX_DIR)))
    analyzer = StandardAnalyzer(Version.LUCENE_CURRENT)
    searcher = IndexSearcher(reader)
    
    query = QueryParser(Version.LUCENE_CURRENT, "page", analyzer).parse("DNA")
    MAX = 1000
    hits = searcher.search(query, MAX)
    
    print hits.scoreDocs

    for hit in hits.scoreDocs:
        print hit.score, hit.doc
    
    #run(searcher, analyzer)
    del searcher