'''
Created on Mar 26, 2018

@author: xica
'''
import pandas as ps
import numpy as np
import csv
import os
from docutils.parsers.rst.directives import encoding
from nltk.corpus import stopwords
import re
from Clustering.KMEANS import Kmeans

def preprocessInputSkill(input):
    output = ''
    stop_words = set(stopwords.words('english'))
    #Keep . '
    prePros = re.sub('[^a-zA-Z0-9 \n\.]', ' ', str(input))
    #collapase whitespaces, and lower case
    prePros = re.sub(r' +', ' ', prePros).lower()
        
    preProsList = prePros.split(' ')
        
    outputList = []
        
    for w in preProsList:
        if w not in stop_words:
            outputList.append(w)
                    
    if len(outputList) > 0:
        output = ' '.join(str(e) for e in outputList)
        output = re.sub('\A\s+', '', output)
    
    return output

def main():
    skills = ps.read_table('../../data/distinct_skill.txt',encoding= 'ISO-8859-1')
    skills_additional = ps.DataFrame(skills.skill.str.split(',')).stack()
    skills_additional = skills_additional.reset_index(drop=True)
    skills_additional = ps.Series.to_frame(skills_additional)
    skills_additional.columns = ['pre_skill_name']
    skills_additional['post_skill_name'] = skills_additional['pre_skill_name'].apply(preprocessInputSkill)
    
    skills_additional.to_csv('../../data/skillMapping.csv', sep='\t')
    
    skills_unique = skills_additional['post_skill_name']
    skills_unique.drop_duplicates(keep='first', inplace = True)
    
    skills_unique_df = skills_unique.to_frame()
    
    kmeanObj = Kmeans()
    
    words = kmeanObj.getWords(inputSet=skills_unique_df['post_skill_name'].values.tolist())
    
    #clusterNumList = [5,10,20,50,100]
    clusterNumList = [50,100]
    #clusterNumList = [5]
    
    for clusterNum in clusterNumList:
        assigned_cluster, cos_sim = kmeanObj.assignClusters(skills_unique_df['post_skill_name'].values.tolist(), clusterNum = clusterNum, words = words, repeatNum = 20)
        cos_sim_df = ps.DataFrame(cos_sim)
        cos_sim_df.to_csv('../../data/cluster_output/cos_sim_%s.csv' % clusterNum, sep='\t')
        print('cos_sim of clusterNum%s is done'%clusterNum)
        skills_unique_df['assigned_cluster_%s'%clusterNum] = assigned_cluster
        skills_unique_df.to_csv('../../data/cluster_output/skills_%s.csv' % clusterNum, sep= '\t')
        print('skills of clusterNum%s is done'%clusterNum)
        cluster_cnt_df = skills_unique_df.groupby(['assigned_cluster_%s'%clusterNum]).agg(['count'])
        cluster_cnt_df.to_csv('../../data/cluster_output/cluster_cnt_%s.csv' % clusterNum, sep='\t')
        print('cluster_cnt of clusterNum%s is done'%clusterNum)
        
if __name__ == '__main__':
    main()