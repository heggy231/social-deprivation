import os
import pandas as pd
import numpy as np
import docx
from collections import defaultdict, Counter
from nltk import word_tokenize, pos_tag, Text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
#from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, f1_score,classification_report, accuracy_score,confusion_matrix
#from ggplot import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import re
import codecs
'''
Opening text files

Bash command to convert everything to UTF-8 and omit invalid characters
for file in *.txt; do iconv -c -t utf-8 "$file" -o "${file%.txt}.utf8.txt"; done
'''
def tokenize(string):
#    string = string.lower()
    
    string = filter(lambda x: x in "abcdefghijklmnopqrstuvwxyz '",string)
    words= word_tokenize(string)
    out = []
    stopwords = ['to','the','a']
    s=SnowballStemmer('english')
    for word in words:
        if word not in stopwords:
            out.append(s.stem(word))
    return out
    
def fpspFilter(nGrams):
    out = []
    for nGram in nGrams:
        for i in fpsp:
            if i in nGram.split(' '):
                out.append(nGram)
    return out

def flatten(l):    
    string = ''
    for s in l:
#        string.join(s)
        string += ' ' + s.encode('ascii','ignore')
    return string#.lower()

def load_corpus(directory):
    texts = {}
    docs = {}
    for f in os.listdir(directory):
        print 'Loading: ', directory+f
        if f.endswith("txt8"):
            with codecs.open(directory+f,'r','ascii','ignore') as text:
                texts[f[:-1]]=text.read()
        elif f.endswith('docx'):
            d = docx.clean(docx.opendocx(directory+f))
            docs[f]=flatten(docx.getdocumenttext(d))#converts to nltk text object
    return texts,docs

fpsp=['i','me','mine','my','myself','myselves']
fppp = ['we','us','ours','our','ourself','ourselves']
tppp = ['they','them','their','theirs','themself','themselves']
    
def before_after(tokens,wordList):
    #takes lowercase
    output = defaultdict(list)
    for i,v in enumerate(tokens):
        if i!=0 and i!=len(tokens)-1:
            if v in wordList:
                output['before '+v].append(tokens[i-1])
                output['after '+v].append(tokens[i+1])               
    return output

def twoGram(string,wordList):
    out=Counter()
    string = string.lower()    
    string = filter(lambda x: x in "abcdefghijklmnopqrstuvwxyz ",string)
    tokens = word_tokenize(string)
    wordCount=len(tokens)
    invWordCount=1/float(wordCount+1)
    for i,token in enumerate(tokens):
        if i!=0 and i!=wordCount-1:
            if token in wordList:
                out[tokens[i-1]+'_'+token]+=invWordCount
                out[token+'_'+ tokens[i+1]]+=invWordCount
    return pd.Series(out)

#def threeGram(string,wordList):
#    out=Counter()
#    string = string.lower()    
#    string = filter(lambda x: x in "abcdefghijklmnopqrstuvwxyz ",string)
#    tokens = word_tokenize(string)
#    wordCount=len(tokens)
#    invWordCount=1/float(wordCount+1)
#    for i,token in enumerate(tokens):
#        if i!=0 and i<wordCount-2:
#            if token in wordList:
#                out[token+'_'+ tokens[i+1]+'_'+tokens[i+2]]+=invWordCount
#    return pd.Series(out)
        
def print_before_after(tokens,wordList):
    for i,v in enumerate(tokens):
        if i!=0 and i!=len(tokens)-1:
            if v in wordList:
                print tokens[i-1], ' ', v, ' ', tokens[i+1]
                
def plot_roc(X, y, clf_class,n_folds=5, **kwargs):
    plt.figure(1,figsize=(12,12))
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    kf = KFold(len(y), n_folds, shuffle=True)
    y_prob = np.zeros((len(y),2))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    for i, (train_index, test_index) in enumerate(kf):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = clf_class(**kwargs)
        clf.fit(X_train,y_train)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y[test_index], y_prob[test_index, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))
        predictions = clf.predict(X_test)
        print 'accuracy: ', accuracy_score(y_test,predictions)
        print classification_report(y_test, predictions)
        print 'confusion matrix: '
        print confusion_matrix(y_test,predictions)
    mean_tpr /= len(kf)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, 'k--',label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)    
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def assess_model(model,xtest,ytest):
    predictions = model.predict(xtest)
#    tp = sum(predictions*ytest)
#    fn = sum((1-predictions)*ytest)
#    tn = sum((1-predictions)*(1-ytest))
#    fp = sum(predictions*(1-ytest))
#    p = sum(ytest)
#    f = sum(1-ytest)
    print 'accuracy: ', accuracy_score(ytest,predictions)
    print classification_report(ytest, predictions)
    print 'confusion matrix: '
    print confusion_matrix(ytest,predictions)
    
def match_filenames(filename,listOfFilenames):
    output = None
    if filename in listOfFilenames:# fix this
        output=filename
    else:
        filename =set(filename)
        for f in listOfFilenames:
            fs = set(str(f))
            if fs.issuperset(filename) or fs.issubset(filename):
                output = f
            else:
                chars = ["'",'_','-','Y','N','.','txt','docx']
                for c in chars:
                    fs.discard(c)
                    filename.discard(c)
                if fs.issuperset(filename) or fs.issubset(filename):
                    output = f         
    return output

def parse_years(s):
    l = len(str(s))
    if (l==3) or (l==4):
        return int(s)
    else:
        return int(s[-4:])
        
if __name__ =='__main__':
    texts,docs =load_corpus("data/allTextData/")
    dtext = defaultdict(str)
    dtext.update(texts)
    dtext.update(docs)
    meta = pd.read_excel('/data/modifiedDeprivedAuthorsTextAnalysis.xls')
    #binarize deprivation
    meta['deprivation']= meta['Deprivation? (Y/N)'].apply(lambda x: x=='Y')
    #make dummy variables for Type of deprivation and Genre, and author?
    meta=meta.join(pd.get_dummies(meta['Type of Deprivation']))
    memoir = lambda x: 'Memoir' if 'Memoir' in x else x #change this to Autobio?
    meta['Genre'] = meta.Genre.apply(memoir)
    meta= meta.join(pd.get_dummies(meta.Genre))
    meta['actualFilename']=meta.Filename.apply(lambda x: match_filenames(x, dtext.keys()))
    meta['text1'] = meta.actualFilename.apply(lambda x: dtext[x])
    meta['year1']=meta['Year Written'].apply(parse_years)
#    meta['fpspGrams']=meta.text1.apply(lambda x: twoGram(x,fpsp))
#    meta['fpppGrams']=mmetaeta.text.apply(lambda x: twoGram(x,fppp))    
    twoGrams=meta.text1.apply(lambda x: twoGram(x,fpsp+fppp))
    threeGrams = meta.text1.apply(lambda x: threeGram(x,['i']))
#    meta.text.apply(lambda x: twoGram(x,fppp))
#    cnt = Counter()
#    for d in meta.twoGrams.values:
#        cnt.update(d)
    #tf-idf
    #using minimum document frequency of 3 gives around 35000 features, and seems reasonable for picking out topics
    tf=TfidfVectorizer(strip_accents='unicode',norm=None,sublinear_tf=1,tokenizer=tokenize,min_df=3)
    tfidf = tf.fit_transform(meta.text1.values)
    features=meta.join(pd.DataFrame(tfidf.todense()))
    features = features.join(twoGrams)
#    meta
    #tf-idf on self referential 2-grams
#    tf2 = TfidfVectorizer(ngram_range=(2,2),strip_accents='unicode',norm=None,sublinear_tf=1,tokenizer=tokenize,min_df=2,vocabulary=fpsp)
#    tfidf2=tf2.fit_transform(meta.text.values)
        #strip unnecessary columns
    y = features.pop('deprivation')
    to_drop=["Prison","Injury","Voluntary",u'Filename', 'actualFilename', u'Author', u'Name of Work',
                  u'Year Written', u'Genre',  u'Deprivation? (Y/N)',u'Type of Deprivation',
                  'WC','WPS','text1']#WPS has many outliars in it and does not seem reliable
    features = features.drop(to_drop,axis=1)
    xtrain,xtest,ytrain,ytest = train_test_split(features,y)
    '''Dimensionality Reduction'''
    p=PCA(n_components=2)
    pcaFeatures=p.fit_transform(features.fillna(0),y)
    
    
    from sklearn.decomposition import TruncatedSVD
    t=TruncatedSVD(n_components=4)#4 features did well with random forest
    truncatedFeatures=t.fit_transform(features.fillna(0),y)  
#    plot_roc(truncatedFeatures,y,LogisticRegression)#does not do well with truncated features
    plot_roc(truncatedFeatures,y,RandomForestClassifier,n_estimators=1000)#this did really well with 2 components
    
    #plot with true labels
    t=TruncatedSVD(n_components=2)
    truncatedFeatures=t.fit_transform(features.fillna(0),y)
    plt.scatter(truncatedFeatures.T[0],truncatedFeatures.T[1],c=y)
    plt.scatter(pcaFeatures.T[0],pcaFeatures.T[1],c=y)
    
#    Boethius has the two outlying points
    
    '''K-means'''    
    from sklearn.cluster import KMeans
    k=KMeans(n_clusters=4)#4 is at an elbow for sse
    km=k.fit_transform(truncatedFeatures) 
    
    
    '''Naive Bayes'''
    #assuming that there is colinearity
#    m = MultinomialNB()
#    m.fit(xtrain,ytrain)
    plot_roc(features.fillna(0),y,MultinomialNB,n_folds=5)
    plot_roc(features.fillna(0),y,BernoulliNB,n_folds=5)
    
    '''Logit'''
    #yields non-singular matrix for genres
    from sklearn.linear_model import LogisticRegression
    plot_roc(features.fillna(0),y,LogisticRegression)
#    plot_roc(features.fillna(0),y,Logit,n_folds=5)
#    xtrain,xtest,ytrain,ytest = train_test_split(features,y)
#    lo = Logit(ytrain,xtrain)
#    result = lo.fit_regularized()
#    fpr, tpr, _ = roc_curve(ytest,result.predict(xtest))
#    from ggplot import *
#    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
#    ggplot(df, aes(x='fpr', y='tpr')) +\
#        geom_line() +\
#        geom_abline(linetype='dashed')
    
    '''Random Forest'''
    from sklearn.ensemble import RandomForestClassifier
    r = RandomForestClassifier(class_weight='auto',n_estimators=1000)
    r.fit(xtrain,ytrain)
    cm = confusion_matrix(ytest,r.predict(xtest))/float(len(xtest))
    accuracy_score(ytest,r.predict(xtest))
    classification_report(ytest,r.predict(xtest))
    plot_roc(features.fillna(0),y,RandomForestClassifier, n_estimators=1000)
    
    '''Gradient Boosting '''
    from sklearn.ensemble import GradientBoostingClassifier
    plot_roc(features.fillna(0),y,GradientBoostingClassifier, n_estimators=100)
    
    '''SVM'''
    from sklearn.svm import SVC
    s = SVC()
    s.fit(xtrain,ytrain)
    s.score(xtest,ytest)
    classification_report(ytest,s.predict(xtest))
    plot_roc(features.fillna(0),y,SVC, probability=True)
    
    '''LDA?'''
    from sklearn.lda import LDA
    l = LDA()
    l.fit(xtrain,ytrain, store_covariance=1)
    accuracy_score(ytest,l.predict(xtest))
    classification_report(ytest,l.predict(xtest))
    plot_roc(features.fillna(0),y,LDA)

#parts of speech

#sentiment


