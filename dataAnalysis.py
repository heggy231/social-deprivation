import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, ward
from loadFiles import *
import plotting as p
import kmeans as kpy
from collections import defaultdict, Counter
from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.linear_model import LogisticRegression
from statsmodels.discrete.discrete_model import Logit
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import (
    cluster,
    datasets,
    decomposition,
    ensemble,
    lda,
    manifold,
    random_projection,
    preprocessing)
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import wordcloud as wc
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import TruncatedSVD, PCA
import networkx as nx
import plotly.plotly as py
from plotly.graph_objs import Data, Heatmap


def tokenize(string):
    #    string = string.lower()
    string = filter(lambda x: x in "abcdefghijklmnopqrstuvwxyz '", string)
    words = word_tokenize(string)
    out = []
    stopwords = ['to', 'the', 'a']
    s = SnowballStemmer('english')
    for word in words:
        if word not in stopwords:
            out.append(s.stem(word))
    return out


def twoGram(string, wordList):
    out = Counter()
    string = string.lower()
    string = filter(lambda x: x in "abcdefghijklmnopqrstuvwxyz ", string)
    tokens = word_tokenize(string)
    wordCount = len(tokens)
    invWordCount = 1 / float(wordCount + 1)
    for i, token in enumerate(tokens):
        if i != 0 and i != wordCount - 1:
            if token in wordList:
                out[tokens[i - 1] + '_' + token] += invWordCount
                out[token + '_' + tokens[i + 1]] += invWordCount
    return pd.Series(out)


def match_filenames(filename, listOfFilenames):
    output = None
    if filename in listOfFilenames:  # fix this
        output = filename
    else:
        filename = set(filename)
        for f in listOfFilenames:
            fs = set(str(f))
            if fs.issuperset(filename) or fs.issubset(filename):
                output = f
            else:
                chars = ["'", '_', '-', 'Y', 'N', '.', 'txt', 'docx']
                for c in chars:
                    fs.discard(c)
                    filename.discard(c)
                if fs.issuperset(filename) or fs.issubset(filename):
                    output = f
    return output


def parse_years(s):
    l = len(str(s))
    if (l == 3) or (l == 4):
        return int(s)
    else:
        return int(s[-4:])


def load_pickeled_features():
    features = pd.read_pickle('data/features_pickle.pkl')
    return features


def make_meta():
    '''turn this into seperate fxns'''
    texts, docs = load_corpus("data/allTextData/")
    dtext = defaultdict(str)
    dtext.update(texts)
    dtext.update(docs)
    meta = pd.read_excel('data/modifiedDeprivedAuthorsTextAnalysis1.xls')
    meta = meta.drop([13, 14], axis=0)
    # binarize deprivation
    meta['deprivation'] = meta['Deprivation? (Y/N)'].apply(lambda x: x == 'Y')
    # make dummy variables for Type of deprivation
    meta = meta.join(pd.get_dummies(meta['Type of Deprivation']))
    memoir = lambda x: 'Memoir' if 'Memoir' in x else x  # change this to Autobio?
    letter = lambda x: 'Letter' if 'Letter' in x else x
    meta['Genre'] = meta.Genre.apply(memoir)
    meta['Genre'] = meta.Genre.apply(letter)
#    meta= meta.join(pd.get_dummies(meta.Genre))
    meta['actualFilename'] = meta.Filename.apply(
        lambda x: match_filenames(x, dtext.keys()))
    meta['text'] = meta.actualFilename.apply(lambda x: dtext[x])
    meta['year'] = meta['Year Written'].apply(parse_years)
    return meta


def make_features(meta):
    # define words of interest
    fpsp = ['i', 'me', 'mine', 'my', 'myself', 'myselves']
    fppp = ['we', 'us', 'ours', 'our', 'ourself', 'ourselves']
    # create dataframe using those words
    twoGrams = meta.text.apply(lambda x: twoGram(x, fpsp + fppp))

    # tf-idf
    # using minimum document frequency of 3 gives around 35000 features, and
    # seems reasonable for picking out topics

    tf = TfidfVectorizer(
        strip_accents='unicode',
        norm=None,
        sublinear_tf=1,
        tokenizer=tokenize,
        min_df=3)
    tfidf = tf.fit_transform(meta.text.values)
    features = meta.join(pd.DataFrame(tfidf.todense()))
    features = features.join(twoGrams)
    return features


def print_features(df, metadf, n, columns=0):
    if not columns:
        columns = metadf.columns
    X_centered = preprocessing.scale(df.fillna(0))
    # 4 features did well with random forest
    t = TruncatedSVD(n_components=n + 1)
    truncatedFeatures = t.fit_transform(X_centered.T)
    for i in xrange(n):
        topic = t.components_[i]
        indx = np.argsort(topic)
        rv_indx = indx[::-1]
        print 'LATENT TOPIC: ', i
        dataframe = metadf[columns].reset_index().loc[rv_indx[:10]]
        print dataframe
    return dataframe


def kcluster(dataframe, n=3, n_clusters=5):
    X_centered = preprocessing.scale(dataframe.fillna(0))
    pca = decomposition.PCA(n_components=n)
    X_pca = pca.fit_transform(X_centered)
    kpy.plot_k_sse(X_pca)
    k = KMeans(n_clusters=n_clusters)
    km = k.fit_transform(X_pca)
    plt.hist(k.labels_)
    return pca, X_pca, k, km


def makeWordcloud(dataframe, **kwargs):
    w = wc.WordCloud(**kwargs)
    w.generate_from_text(dataframe.text.sum())
    plt.imshow(w)
    return w


def heatmap(df):
    '''Plot heatmap of input dataframe using plotly'''
    for col in df.columns:
        df[col] /= df[col].max()
    data = Data([Heatmap(z=df.values)])
    plot_url = py.plot(data, filename='basic-heatmap')
    print 'heatmap url:', plot_url


def knn(df, axis=None, labels=None):
    dist = 1 - cosine_similarity(df.values)
    # define the linkage_matrix using ward clustering pre-computed distances
    linkage_matrix = ward(dist)

    fig, ax = plt.subplots(figsize=(15, 20))  # set size
    ax = dendrogram(linkage_matrix, orientation="right", labels=labels)

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')

    plt.tight_layout()

if __name__ == '__main__':
    meta = make_meta()
    features = load_pickeled_features()
    to_drop = [
        "Prison",
        "Injury",
        "Voluntary",
        u'Filename',
        'actualFilename',
        u'Author',
        u'Name of Work',
        u'Year Written',
        u'Genre',
        u'Deprivation? (Y/N)',
        u'Type of Deprivation',
        'WC',
        'WPS',
        'text1']  # 'text','year'. WPS has many outliers in it and does not seem reliable
    features = features.drop(to_drop, axis=1)
    # dropping boethius since it seems to be so unique
    features = features.drop([13, 14], axis=0)
    features = features.reset_index()
    y = features.pop('deprivation')
    y1 = y.reset_index(drop=1)
    justDFeatures = features[y1 == 1]
    nDFeatures = features[y1 == 0]
    justDmeta = meta[meta.deprivation == 1]
    nDmeta = meta[meta.deprivation == 0]

    liwc = ['funct', u'pronoun', u'ppron', u'i',
            u'we', u'you', u'shehe', u'they',
            u'ipron', u'article', u'past', u'present',
            u'future', u'social', u'family', u'friend',
            u'humans', u'affect', u'posemo', u'negemo',
            u'anx', u'anger', u'sad', u'cogmech',
            u'insight', u'certain', u'percept', u'see',
            u'hear', u'feel', u'bio', u'body',
            u'health', u'sexual', u'ingest', u'space',
            u'time', u'work', u'achieve', u'leisure',
            u'home', u'death']
    '''Graph'''
    G = nx.Graph(data=cosine_similarity(meta[liwc]))

    '''Dimensionality Reduction'''
    p = PCA(n_components=3)
    pcaFeatures = p.fit_transform(features.fillna(0), y)

    t = TruncatedSVD(n_components=10)  # 4 features did well with random forest
    truncatedFeatures = t.fit_transform(justDFeatures.fillna(0), y)

    cols = ['Author', 'Genre', 'year', 'Name of Work',
            'deprivation', 'Type of Deprivation', u'funct', u'pronoun',
            u'ppron', u'i', u'we']
    sw = stopwords.words('english') + ['one', 'would']
    makeWordcloud(
        print_features(
            justDFeatures,
            justDmeta,
            1),
        stopwords=sw,
        ranks_only=1,
        width=800,
        height=400)
    makeWordcloud(
        print_features(
            nDFeatures,
            nDmeta,
            1),
        stopwords=sw,
        ranks_only=1,
        width=800,
        height=400)
    featured_documents = {}
    for i in range(10):
        featured_documents[i] = np.argsort(umatrix[i])[:10]
    justDmeta.reset_index().loc[featured_documents[4]]
#    Boethius has the two outlying points
    '''KNN'''
    links = linkage(meta[liwc].values)
    dendrogram(links)
    knn(meta[liwc].T, labels=liwc)
    knn(meta[liwc], labels=meta['Name of Work'].values)
    '''K-means'''
    k = KMeans(n_clusters=5)  # 5 is at an elbow for sse in 2-d
    km = k.fit_transform(truncatedFeatures)

    '''PCA'''
    pca, X_pca, k, km = kcluster(justDFeatures, n_clusters=8)
    print features.columns[np.argsort(pca.components_[0])[:100]]

#    plt.savefig("scree.png", dpi= 100)

    pca = decomposition.PCA(n_components=2)
    X_pca = pca.fit_transform(X_centered)
    plot_embedding(X_pca, y)
    k.plot_k_sse(X_pca)  # for 2 components 5 clusters

    ''' Supervised Learning'''
    # Logistic Regression and Random Forest seem to perform the best
    # Nonfiction seems unpredictable, while fiction, letters and poetry
    # are somewhat predictabe
    for genre in set(meta.Genre):
        df = meta[meta.Genre == genre].reset_index()
        if len(df) > 20:
            y = df.pop('deprivation')
            print genre, 'Logit'
            p.plot_roc(df[liwc].fillna(0), y, LogisticRegression)
            print genre, 'Random Forest'
            p.plot_roc(df[liwc].fillna(0), y, RandomForestClassifier)
