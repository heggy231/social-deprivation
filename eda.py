# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/d/.spyder2/.temp.py
"""
import os
import pandas as pd
import numpy as np
import docx
from collections import defaultdict
from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve





'''
Opening text files

Bash command to convert everything to UTF-8 and omit invalid characters
for file in *.txt; do iconv -c -t utf-8 "$file" -o "${file%.txt}.utf8.txt"; done
'''
def tokenize(string):
    string = string.lower()
    string = filter(lambda x: x in "abcdefghijklmnopqrstuvwxyz .'",string)
    return word_tokenize(string)


#class corpus(object):
def flatten(l):
    string = ''
    for s in l:
#        string.join(s)
        string += ' ' + s.encode('ascii', 'ignore')#.decode()
    return string
    
def load_corpus(directory):
    texts = {}
    docs = {}
    for f in os.listdir(directory):
        print 'loading ', directory+f
        if f.endswith("txt8"):
            with open(directory+f,'r') as text:
                texts[f[:-1]]=text.read()#.decode()
        elif f.endswith('docx'):
            d = docx.opendocx(directory+f)
            docs[f]=flatten(docx.getdocumenttext(d))
    return texts, docs

#open docx
#def load_docx(directory):
#d=docx.opendocx('data/allTextData/docx/Brown-Letter_dated_November_16-1859-Y.docx')
#dtext = docx.getdocumenttext(d) #returns list of paragraphs

#using existing features
correlation = meta.corr().deprivation
#tf-ifd

#parts of speech

#sentiment


if __name__ =='__main__':
    texts,docs =load_corpus("data/allTextData/")
    mfl = [x.lower() for x in meta.Filename.values]
    afl = [x.lower() for x in texts.keys()]
    
#    stringord = lambda a: [ord(c) for c in a]
#    import re
#    strip = lambda x: re.sub(r'\W+', '', x)
    #strip more stuff
    
#exploratory
tokens = [w.lower() for w in word_tokenize(texts['Berkman-Prison_Memoirs_of_an_Anarchist-1912-Y.txt'])]

fpsp=['i','me','mine','my','myself','myselves']

def count_fpsp(tokens):
    return [(tokens.count(i),i) for i in fpsp]
    

def before_after(tokens,wordList):
    #takes lowercase
    output = defaultdict(list)
    for i,v in enumerate(tokens):
        if i!=0 and i!=len(tokens)-1:
            if v in wordList:
                output['before '+v].append(tokens[i-1])
                output['after '+v].append(tokens[i+1])               
    return output

"""
Model Building
Logit using statsmodels
"""
'''
Import metadata and files

Note: 'pd.read_table' doesn't work so well
'''

meta = pd.read_excel('/data/modifiedDeprivedAuthorsTextAnalysis.xls')
#binarize deprivation

meta['deprivation']= meta['Deprivation? (Y/N)'].apply(lambda x: x=='Y')
#make dummy variables for Type of deprivation and Genre, and author?
meta=meta.join(pd.get_dummies(meta['Type of Deprivation']))
memoir = lambda x: 'Memoir' if 'Memoir' in x else x
meta.Genre.apply(memoir)
meta= meta.join(pd.get_dummies(meta.Genre))
y = meta.pop('deprivation')
to_drop=["Prison","Injury","Voluntary",u'Filename',              u'Author',        u'Name of Work',
              u'Year Written',               u'Genre',  u'Deprivation? (Y/N)',
       u'Type of Deprivation']
features = meta.drop(to_drop,axis=1)
features['intercept']=1
#features['guesses']= features.i * 
xtrain,xtest,ytrain,ytest = train_test_split(features,y)
lo = Logit(ytrain,xtrain)
result = lo.fit_regularized()
fpr, tpr, _ = roc_curve(ytest,result.predict(xtest))
from ggplot import *
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')