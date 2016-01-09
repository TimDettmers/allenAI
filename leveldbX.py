'''
Created on Nov 25, 2015

@author: tim
'''
import leveldb
import simplejson
from Queue import Queue
from os.path import join, exists
import urllib2
from threading import Thread
from os.path import expanduser
from dateutil import parser
home = expanduser("~")



class Table(object):
    def __init__(self, name, path, ip, hasServer):
        self.name = name
        self.queue = Queue()
        self.hasServer = hasServer
        self.db = leveldb.LevelDB(join(path,name))
        self.ip = ip
        
    def get(self, key):
        ret = None
        try:
            ret = simplejson.loads(self.db.Get(join(self.name, key)))
        except KeyError:
            ret = None
            
        if self.hasServer and ret == None: 
            r = urllib2.urlopen(join(self.ip, self.name, key))
            value = simplejson.load(r)
            self.set(key, value)
            return value
        else:
            return ret
        
    
    def set(self, key, value):
        self.db.Put(join(self.name,key),simplejson.dumps(value))
        
    def post(self, key, value):
        if self.hasServer:
            req = urllib2.Request(join(self.ip, self.name, key))
            req.add_header('Content-Type', 'application/json')
            print simplejson.dumps(value)
            if isinstance(value, basestring):
                response = urllib2.urlopen(req, value)
            else:
                response = urllib2.urlopen(req, simplejson.dumps(value))
            
            self.set(key, value)
        
        
    def delete(self, key):
        self.db.Delete(join(self.name,key))
        
        
    def scan(self):
        for key, value in self.db.RangeIter():
            yield key, simplejson.loads(value)
        
        


class LevelDBX(object):
    def __init__(self, isServer = False, path=join(home, '.nlpdb'), ip='http://127.0.0.1:5000'):
        self.isServer = isServer
        self.ip = ip
        
        if isServer:
            self.path = join(home,'.nlpdb_server')     
        else:
            self.path  = path
            
        self.db = leveldb.LevelDB(self.path)
        
        try:
            stamp = parser.parse(simplejson.load(urllib2.urlopen(join(self.ip, 'ping'))))
        except:
            stamp = None
        if stamp != None: self.hasServer = True
        else: self.hasServer = False
        
    def join(self):
        if self.isServer:
            self.bgServer.join()
        
    def get_table(self, name):
        return Table(name, self.path, self.ip, self.hasServer)
    
    def table_exists(self, name):
        return exists(join(self.path, name))




