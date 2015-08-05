# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
/home/d/.spyder2/.temp.py
"""
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
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

meta = pd.read_excel('/data/modifiedDeprivedAuthorsTextAnalysis.xls')
#binarize deprivation

meta['deprivation']= meta['Deprivation? (Y/N)'].apply(lambda x: x=='Y')
meta.drop(u'Deprivation? (Y/N)',axis=1)
#make dummy variables for Type of deprivation and Genre, and author?
meta=meta.join(pd.get_dummies(meta['Type of Deprivation']))

#get years?

'''
Opening text files

Bash command to convert everything to UTF-8 and omit invalid characters
for file in *.txt; do iconv -c -t utf-8 "$file" -o "${file%.txt}.utf8.txt"; done
'''


f= 'Bakunin-Confession_to_Tsar_Nicholas_I-1851-Nonfiction-Y-Prison.utf8.txt'
with open(f,'r') as ex:
    text = ex.read()

texts = {}
#texts = pd.DataFrame()
for f in os.listdir("."):
    if f.endswith("utf8.txt"):
        with open(f,'r') as text:
            texts[f]=text.read().decode()

#open docx
d=docx.opendocx('data/allTextData/Brown-Letter_dated_November_16-1859-Y.docx')
dtext = docx.getdocumenttext(d) #returns list of paragraphs

#using existing features
correlation = meta.corr().deprivation
#tf-ifd

#parts of speech

#sentiment
