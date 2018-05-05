'''
Created on Mar 26, 2018

@author: xica
'''
import sys
import numpy
from nltk.cluster import KMeansClusterer, GAAClusterer, euclidean_distance, cosine_distance
import nltk.corpus
from nltk import decorators
import nltk.stem
import scipy as sp
from scipy import spatial

class Kmeans():
    
    def normalize_word(self, word):
        stemmer_func = nltk.stem.snowball.EnglishStemmer().stem
        return stemmer_func(word.lower())
    
    def getWords(self, inputSet):
        words = set()

        for title in inputSet:

            for word in title.split():

                words.add(self.normalize_word(word))

        return list(words)
    
    def vectorspaced(self,input, words):
        
        stopwords = set(nltk.corpus.stopwords.words('english'))

        input_components = [self.normalize_word(word) for word in input.split()]

        return numpy.array([

            word in input_components and not word in stopwords

                for word in words], numpy.short)    
        
    def assignClusters(self, inputSet, clusterNum, words, distance = cosine_distance, repeatNum = 10):
        clusters = KMeansClusterer(clusterNum, distance, repeatNum)
        assigned_clusters = clusters.cluster([self.vectorspaced(input, words) for input in inputSet], assign_clusters=True)            
        cos_sim = self.getCosSim(clusters, clusterNum)
            
        return assigned_clusters, cos_sim
    
    def getCosSim(self, clusters, clusterNum):
        cos_sim = numpy.zeros((clusterNum,clusterNum))
        for i in range(0,clusterNum):
            if(numpy.dot(clusters.means()[i],clusters.means()[i]) == 0):
                cos_sim = 0
            for j in range(0,clusterNum):  
                if(numpy.dot(clusters.means()[j],clusters.means()[j]) == 0):
                    cos_sim = 0
                else:
                    cos_sim[i][j] = 1-spatial.distance.cosine(clusters.means()[i],clusters.means()[j])
        return cos_sim