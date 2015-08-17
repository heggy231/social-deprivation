# -*- coding: utf-8 -*-
'''
Opening text files

Bash command to convert everything to UTF-8 and omit invalid characters
for file in *.txt; do iconv -c -t utf-8 "$file" -o "${file%.txt}.utf8.txt"; done
'''
import os
import docx
import codecs


def flatten(l):
    string = ''
    for s in l:
        #        string.join(s)
        string += ' ' + s.encode('ascii', 'ignore')
    return string  # .lower()


def load_corpus(directory):
    texts = {}
    docs = {}
    for f in os.listdir(directory):
        print 'Loading: ', directory + f
        if f.endswith("txt8"):
            with codecs.open(directory + f, 'r', 'ascii', 'ignore') as text:
                texts[f[:-1]] = text.read()
        elif f.endswith('docx'):
            d = docx.clean(docx.opendocx(directory + f))
            # converts to nltk text object
            docs[f] = flatten(docx.getdocumenttext(d))
    return texts, docs
