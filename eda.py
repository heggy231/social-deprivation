# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/d/.spyder2/.temp.py
"""

import pandas as pd
import numpy as np
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#from sklearn.preprocessing import normalize
#from sklearn.metrics.pairwise import cosine_similarity
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.wordnet import WordNetLemmatizer

'''
Import metadata and files

Note: 'pd.read_table' doesn't work so well
'''

meta = pd.read_excel('/data/DeprivedAuthorsTextAnalysis.xls')
#binarize deprivation

meta['deprivation']= meta['Deprivation? (Y/N)'].apply(lambda x: x=='Y')
meta.drop(u'Deprivation? (Y/N)',axis=1)
#make dummy variables for Type of deprivation and Genre

#open txts
f= 'data/allTextData/Blanqui-Textes_Choisi-1971-Nonfiction-N.txt'
with open(f,'r') as ex:
    text = ex.read()
    
#open docx
d=docx.opendocx('data/allTextData/Brown-Letter_dated_November_16-1859-Y.docx')
dtext = docx.getdocumenttext(d)

#using ex

#tf-ifd

#parts of speech

#sentiment
