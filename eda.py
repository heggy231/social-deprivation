import os
import pandas as pd
import numpy as np
import docx
from collections import defaultdict
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
    string = string.lower()
    
    string = filter(lambda x: x in "abcdefghijklmnopqrstuvwxyz '",string)
    words= word_tokenize(string)
    out = []
    stopwords = ['to','the','a']
    s=SnowballStemmer('english')
    for word in words:
        if word not in stopwords:
            out.append(s.stem(word))
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
    return texts, docs

fpsp=['i','me','mine','my','myself','myselves']
fppp = ['we','us','ours','our','ourself','ourselves']

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
#    predictions = clf.predict(X)
#    print 'accuracy: ', accuracy_score(y,predictions)
#    print classification_report(y, predictions)
#    print 'confusion matrix: '
#    print confusion_matrix(y,predictions)
    
def assess_model(model,xtest,ytest):
    predictions = model.predict(xtest)
    tp = sum(predictions*ytest)
    fn = sum((1-predictions)*ytest)
    tn = sum((1-predictions)*(1-ytest))
    fp = sum(predictions*(1-ytest))
    p = sum(ytest)
    f = sum(1-ytest)
    print 'accuracy: ', accuracy_score(ytest,predictions)
    print classification_report(ytest, predictions)
    print 'confusion matrix: '
    print confusion_matrix(ytest,predictions)
    
def match_filenames(filename,listOfFilenames):
    output = None
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
    meta['text'] = meta.actualFilename.apply(lambda x: dtext[x])

    #using existing features
    correlation = meta.corr().deprivation
    

    #features['intercept']=1
    
    #tf-idf
    #using minimum document frequency of 3 gives around 35000 features, and seems reasonable for picking out topics
    tf=TfidfVectorizer(strip_accents='unicode',norm=None,sublinear_tf=1,tokenizer=tokenize,min_df=3)
    
    meta=meta.join(pd.DataFrame(tf.fit_transform(meta.text.values).todense()))
#    #tf-idf on self referential 2-grams
#    tf2 = TfidfVectorizer(strip_accents='unicode',norm=None,sublinear_tf=1,tokenizer=tokenize,min_df=2,vocabulary=fpsp)
#    tfidf2=tf2.fit_transform(meta.text.values)
        #strip unnecessary columns
    y = meta.pop('deprivation')
    to_drop=["Prison","Injury","Voluntary",u'Filename', 'actualFilename', u'Author', u'Name of Work',
                  u'Year Written', u'Genre',  u'Deprivation? (Y/N)',u'Type of Deprivation',
                  'WC','text']
    features = meta.drop(to_drop,axis=1)
    xtrain,xtest,ytrain,ytest = train_test_split(features,y)

    '''Naive Bayes'''
    #fill in assumptions here
    m = MultinomialNB()
    m.fit(xtrain,ytrain)
    
    b= BernoulliNB()
    
    '''Logit'''
    #yields non-singular matrix for genres
    lo = Logit(ytrain,xtrain)
    result = lo.fit_regularized()
    fpr, tpr, _ = roc_curve(ytest,result.predict(xtest))
    from ggplot import *
    df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
    ggplot(df, aes(x='fpr', y='tpr')) +\
        geom_line() +\
        geom_abline(linetype='dashed')
    
    '''Random Forest'''
    from sklearn.ensemble import RandomForestClassifier
    r = RandomForestClassifier(class_weight='auto',n_estimators=1000)
    r.fit(xtrain,ytrain)
    cm = confusion_matrix(ytest,r.predict(xtest))/float(len(xtest))
    accuracy_score(ytest,r.predict(xtest))
    classification_report(ytest,r.predict(xtest))
    plot_roc(features,y,RandomForestClassifier, n_estimators=1000)
    
    '''Gradient Boosting '''
    from sklearn.ensemble import GradientBoostingClassifier
    plot_roc(features,y,GradientBoostingClassifier,n_folds=3, n_estimators=100)
    
    '''SVM'''
    from sklearn.svm import SVC
    s = SVC()
    s.fit(xtrain,ytrain)
    s.score(xtest,ytest)
    classification_report(ytest,s.predict(xtest))
    plot_roc(features,y,SVC, probability=True)
    
    '''LDA?'''
    from sklearn.lda import LDA
    l = LDA()
    l.fit(xtrain,ytrain, store_covariance=1)
    accuracy_score(ytest,l.predict(xtest))
    classification_report(ytest,l.predict(xtest))
    plot_roc(features,y,LDA)

#parts of speech

#sentiment


