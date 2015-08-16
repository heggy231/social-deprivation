
import pandas as pd
import numpy as np
#import loadFiles as lf
from loadFiles import *
import plotting as p
import kmeans as kpy
from collections import defaultdict, Counter
from nltk import word_tokenize, pos_tag, Text
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, f1_score,classification_report, accuracy_score,confusion_matrix
#from ggplot import *
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn import (cluster, datasets, decomposition, ensemble, lda, manifold, random_projection, preprocessing)
from sklearn.lda import LDA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import wordcloud as wc
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, PCA   
import networkx as nx


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

def load_pickeled_features():
    features = pd.read_pickle('data/features_pickle.pkl')
    return features
    
def make_meta():
    '''turn this into seperate fxns'''
    texts,docs =load_corpus("data/allTextData/")
    dtext = defaultdict(str)
    dtext.update(texts)
    dtext.update(docs)
    meta = pd.read_excel('data/modifiedDeprivedAuthorsTextAnalysis1.xls')
    meta=meta.drop([13,14],axis=0)
    #binarize deprivation
    meta['deprivation']= meta['Deprivation? (Y/N)'].apply(lambda x: x=='Y')
    #make dummy variables for Type of deprivation and Genre, and author?
    meta=meta.join(pd.get_dummies(meta['Type of Deprivation']))
    memoir = lambda x: 'Memoir' if 'Memoir' in x else x #change this to Autobio?
    meta['Genre'] = meta.Genre.apply(memoir)
#    meta= meta.join(pd.get_dummies(meta.Genre))
    meta['actualFilename']=meta.Filename.apply(lambda x: match_filenames(x, dtext.keys()))
    meta['text'] = meta.actualFilename.apply(lambda x: dtext[x])
    meta['year']=meta['Year Written'].apply(parse_years)
    return meta

def make_features(meta):   
    #define words of interest
    fpsp=['i','me','mine','my','myself','myselves']
    fppp = ['we','us','ours','our','ourself','ourselves']
    #create dataframe using those words
    twoGrams=meta.text.apply(lambda x: twoGram(x,fpsp+fppp))

    #tf-idf
    #using minimum document frequency of 3 gives around 35000 features, and seems reasonable for picking out topics
    
    tf=TfidfVectorizer(strip_accents='unicode',norm=None,sublinear_tf=1,tokenizer=tokenize,min_df=3)
    tfidf = tf.fit_transform(meta.text.values)
    features=meta.join(pd.DataFrame(tfidf.todense()))
    features = features.join(twoGrams)
    return features
    
def print_features(df,metadf, n, columns=0):
    if not columns:
        columns=metadf.columns
    X_centered = preprocessing.scale(df.fillna(0))
    t=TruncatedSVD(n_components=n+1)#4 features did well with random forest
    truncatedFeatures=t.fit_transform(X_centered.T)
    for i in xrange(n):
        topic=t.components_[i]
        indx = np.argsort(topic)
        rv_indx = indx[::-1]  
        print 'LATENT TOPIC: ',i
        dataframe=metadf[columns].reset_index().loc[rv_indx[:10]]
        print dataframe
    return dataframe

def kcluster(dataframe, n=3, n_clusters=5):
    X_centered = preprocessing.scale(dataframe.fillna(0))
    pca = decomposition.PCA(n_components=n)
    X_pca = pca.fit_transform(X_centered)
    kpy.plot_k_sse(X_pca)
    k=KMeans(n_clusters=n_clusters)#5 is at an elbow for sse in 2-d
    km=k.fit_transform(X_pca)
    plt.hist(k.labels_)
#        scree_plot(n, pca)
    return pca, X_pca, k, km    
    
def makeWordcloud(dataframe,**kwargs):
    w=wc.WordCloud(**kwargs)
    w.generate_from_text(dataframe.text.sum())
    plt.imshow(w)
    return w
    
if __name__ =='__main__':
    meta=make_meta()
    features = load_pickeled_features()
    to_drop=["Prison","Injury","Voluntary",u'Filename', 'actualFilename', u'Author', u'Name of Work',
                  u'Year Written', u'Genre',  u'Deprivation? (Y/N)',u'Type of Deprivation',
                  'WC','WPS','text1']#'text','year'. WPS has many outliers in it and does not seem reliable
    features = features.drop(to_drop,axis=1)
    features=features.drop([13,14],axis=0) #dropping boethius since it seems to be so unique
    features=features.reset_index()   
    y = features.pop('deprivation')
    y1=y.reset_index(drop=1)
    justDFeatures = features[y1==1]
    nDFeatures = features[y1==0]    
    justDmeta   = meta[meta.deprivation==1] 
    nDmeta   = meta[meta.deprivation==0] 
    
    liwc = ['funct',
 u'pronoun',
 u'ppron',
 u'i',
 u'we',
 u'you',
 u'shehe',
 u'they',
 u'ipron',
 u'article',
 u'past',
 u'present',
 u'future',
 u'social',
 u'family',
 u'friend',
 u'humans',
 u'affect',
 u'posemo',
 u'negemo',
 u'anx',
 u'anger',
 u'sad',
 u'cogmech',
 u'insight',
 u'certain',
 u'percept',
 u'see',
 u'hear',
 u'feel',
 u'bio',
 u'body',
 u'health',
 u'sexual',
 u'ingest',
 u'space',
 u'time',
 u'work',
 u'achieve',
 u'leisure',
 u'home',
 u'death']
    '''Graph'''
    G=nx.Graph(data=cosine_similarity(meta[liwc]))
         
    '''Dimensionality Reduction'''
    p=PCA(n_components=3)
    pcaFeatures=p.fit_transform(features.fillna(0),y)
    
    
    t=TruncatedSVD(n_components=10)#4 features did well with random forest
    truncatedFeatures=t.fit_transform(justDFeatures.fillna(0),y)  
##    plot_roc(truncatedFeatures,y,LogisticRegression)#does not do well with truncated features
#    plot_roc(truncatedFeatures,y,RandomForestClassifier,n_estimators=1000)#does well with 4 features
#    #plot with true labels
#    t=TruncatedSVD(n_components=2)
#    truncatedFeatures=t.fit_transform(features.fillna(0),y)
#    plt.scatter(truncatedFeatures.T[0],truncatedFeatures.T[1],c=y)
#    plt.scatter(pcaFeatures.T[0],pcaFeatures.T[1],c=y)
    
    cols = ['Author',
     'Genre',
     'year',
     'Name of Work',
     'deprivation',
     'Type of Deprivation',
     u'funct',
     u'pronoun',
     u'ppron',
     u'i',
     u'we']
    sw = stopwords.words('english')+['one','would']
    makeWordcloud(print_features(justDFeatures,justDmeta,1), stopwords=sw,ranks_only=1,width=800,height=400)
    makeWordcloud(print_features(nDFeatures,nDmeta,1), stopwords=sw,ranks_only=1,width=800,height=400)
    featured_documents = {}
    for i in range(10):
        featured_documents[i]=np.argsort(umatrix[i])[:10]
    justDmeta.reset_index().loc[featured_documents[4]]
#    Boethius has the two outlying points
    
    '''K-means'''    
    k=KMeans(n_clusters=5)#5 is at an elbow for sse in 2-d
    km=k.fit_transform(truncatedFeatures) 
    
    pca, X_pca, k, km = kcluster(justDFeatures,n_clusters=8)
    print features.columns[np.argsort(pca.components_[0])[:100]]
    
    #fix the following    
    mostProb = np.argsort(r.predict_proba(features.fillna(0)).T[0])[:5]
#    plt.savefig("scree.png", dpi= 100)


    pca = decomposition.PCA(n_components=2)
    X_pca = pca.fit_transform(X_centered)
    plot_embedding(X_pca, y)
    k.plot_k_sse(X_pca) #for 2 components 5 clusters

    ''' Supervised Learning'''
    
    xtrain,xtest,ytrain,ytest = train_test_split(features,y1)
    '''Naive Bayes'''
    #assuming that there is colinearity
#    m = MultinomialNB()
#    m.fit(xtrain,ytrain)
    plot_roc(features.fillna(0),y,MultinomialNB,n_folds=5)
    plot_roc(features.fillna(0),y,BernoulliNB,n_folds=5)
    plot_roc(truncatedFeatures,y,MultinomialNB,n_folds=5)
    plot_roc(truncatedFeatures,y,BernoulliNB,n_folds=5)#performs horrbily
    '''Logit'''
    #Nonfiction seems unpredictable, while fiction is somewhat predictable letters
    for genre in set(meta.Genre):
        df=meta[meta.Genre==genre].reset_index()
        if len(df)>10:
            y=df.pop('deprivation')
            print genre
#            p.plot_roc(df[liwc].fillna(0),y,LogisticRegression)
#            p.plot_roc(df[liwc].fillna(0),y,RandomForestClassifier)
            
      
    '''Random Forest'''
    r = RandomForestClassifier(class_weight='auto',n_estimators=1000)
    r.fit(xtrain.fillna(0),ytrain)
    cm = confusion_matrix(ytest,r.predict(xtest.fillna(0)))/float(len(xtest))
    accuracy_score(ytest,r.predict(xtest.fillna(0)))
    classification_report(ytest,r.predict(xtest.fillna(0)))
    plot_roc(features.fillna(0),y,RandomForestClassifier, n_estimators=1000)
    
    '''Gradient Boosting '''
    plot_roc(features.fillna(0),y,GradientBoostingClassifier, n_estimators=100)
    
    '''SVM'''
    s = SVC()
    s.fit(xtrain,ytrain)
    s.score(xtest,ytest)
    classification_report(ytest,s.predict(xtest))
    plot_roc(features.fillna(0),y,SVC, probability=True)
    
    '''LDA?'''
    l = LDA()
    l.fit(xtrain,ytrain, store_covariance=1)
    accuracy_score(ytest,l.predict(xtest))
    classification_report(ytest,l.predict(xtest))
    plot_roc(features.fillna(0),y,LDA)
    
    
    '''lmer'''
    import statsmodels.api as sm 
    import statsmodels.formula.api as smf
    
    #data = sm.datasets.get_rdataset("dietox", "geepack").data
    
    md = smf.mixedlm("i ~ deprivation", meta, groups=meta["Genre"]) 
    mdf = md.fit() 
    
    print mdf.summary()
    #parts of speech
    