import os
import pandas as pd
#import numpy as np
import docx
from collections import defaultdict
from nltk import word_tokenize, pos_tag, Text
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, 
#from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
#from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, f1_score,classification_report, accuracy_score
#from ggplot import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.cross_validation import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

'''
Opening text files

Bash command to convert everything to UTF-8 and omit invalid characters
for file in *.txt; do iconv -c -t utf-8 "$file" -o "${file%.txt}.utf8.txt"; done
'''
#def tokenize(string):
#    string = string.lower()
#    string = filter(lambda x: x in "abcdefghijklmnopqrstuvwxyz '",string)
#    return word_tokenize(string)

#class corpus(object):
def flatten(l):
    string = ''
    for s in l:
#        string.join(s)
        string += ' ' + s.encode('UTF-8', 'ignore')#.decode()
    return string.lower()
    
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
    predictions = clf.predict(X)
    print 'accuracy: ', accuracy_score(y,predictions)
    print classification_report(y, predictions)
    print 'confusion matrix: '
    print confusion_matrix(y,predictions)
    
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
    print confusion_matrix(ytest,r.predict(xtest))
    
def match_filenames(filename,listOfFilenames):
    extra_words = ['Injury',              u'Prison',
                 u'Voluntary',               u'Essay',             u'Fiction',
                    u'Letter',             u'Letters',              u'Lyrics',
                 u'MathPaper',          u'Meditation',              u'Memoir',
                u'Nonfiction',                u'Play',              u'Poetry',
                    u'Quotes',              u'Speech']
    output = None
    filename =set(filename)
    for f in listOfFilenames:
        fs = set(f)
        if fs.issuperset(filename) or fs.issubset(filename):
            output = f
#        else:
#            for word in extra_words:
#                stripped=filename.strip(word)
#                if (f in stripped) or (stripped in f):
#                    output = f
#    if output==None:
#        for f in listOfFilenames:
#            some clever regex            
    return output
    
def vectorize(string,**kwargs):
    c = CountVectorizer(decode_error='replace',strip_accents='unicode')
    d=c.fit_transform(string.split(' ')).todense()
    z = TfidfTransformer(**kwargs)#norm=None,smooth_idf=True,sublinear_tf=True
    return z.fit_transform(d).todense()
    
    
if __name__ =='__main__':
    texts,docs =load_corpus("data/allTextData/")
#    mfl = [x.lower() for x in meta.Filename.values]
#    afl = [x.lower() for x in texts.keys()]
#    filter(isalnum,mfl)
#    matching_files=[x in texts.keys() for x in meta.Filename.values]
#    
#    missing_files = meta[map(lambda x: not x,matching_files)].Filename
    dtext = defaultdict(str)
    dtext.update(texts)
    dtext.update(docs)
    meta = pd.read_excel('/data/modifiedDeprivedAuthorsTextAnalysis.xls')
    #binarize deprivation
    meta['deprivation']= meta['Deprivation? (Y/N)'].apply(lambda x: x=='Y')
    #make dummy variables for Type of deprivation and Genre, and author?
    meta=meta.join(pd.get_dummies(meta['Type of Deprivation']))
    memoir = lambda x: 'Memoir' if 'Memoir' in x else x #fix this
    meta['Genre'] = meta.Genre.apply(memoir)
    meta= meta.join(pd.get_dummies(meta.Genre))
    meta['actualFilename']=meta.Filename.apply(lambda x: match_filenames(x, dtext.keys()))
    meta['text'] = meta.actualFilename.apply(lambda x: dtext[x])
    #    stringord = lambda a: [ord(c) for c in a]
    #    import re
    #    strip = lambda x: re.sub(r'\W+', '', x)
        #strip more stuff
    
    #vectorize and tf-idf

    
    #exploratory
    tokens = [w.lower() for w in word_tokenize(texts['Berkman-Prison_Memoirs_of_an_Anarchist-1912-Y.txt'])]
    
    #using existing features
    correlation = meta.corr().deprivation
    
    
    y = meta.pop('deprivation')
    to_drop=["Prison","Injury","Voluntary",u'Filename', u'Author', u'Name of Work',
                  u'Year Written', u'Genre',  u'Deprivation? (Y/N)',u'Type of Deprivation',
                  'WC']
    features = meta.drop(to_drop,axis=1)
    #features['intercept']=1
    #features['guesses']= features.i * 
    xtrain,xtest,ytrain,ytest = train_test_split(features,y)
    
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

#open docx
#def load_docx(directory):
#d=docx.opendocx('data/allTextData/docx/Brown-Letter_dated_November_16-1859-Y.docx')
#dtext = docx.getdocumenttext(d) #returns list of paragraphs

#tf-ifd

#parts of speech

#sentiment


