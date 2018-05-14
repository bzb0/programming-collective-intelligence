import re

import feedparser
import numpy as np


def stripHTML(h):
    p = ''
    s = 0
    for c in h:
        if c == '<':
            s = 1
        elif c == '>':
            s = 0
            p += ' '
        elif s == 0:
            p += c
    return p


def separatewords(text):
    # splitter = re.compile("(\w[\w']*\w|\w)")
    # return [s.lower() for s in splitter.split(text) if len(s) > 3]
    return [word.lower() for word in re.findall(r'\b\w+\b', text) if len(word) > 3]


def getarticlewords(feedlist):
    allwords = {}
    articlewords = []
    articletitles = []
    articlecounter = 0

    # Loop over every feed
    for feed in feedlist:
        f = feedparser.parse(feed)

        # Loop over every article
        for entry in f.entries:
            # Ignore identical articles
            if entry.title in articletitles:
                continue

            # Extract the words
            txt = entry.title + stripHTML(entry.description)
            words = separatewords(txt)
            articlewords.append({})
            articletitles.append(entry.title)

            # Increase the counts for this word in allwords and in articlewords
            for word in words:
                allwords.setdefault(word, 0)
                allwords[word] += 1
                articlewords[articlecounter].setdefault(word, 0)
                articlewords[articlecounter][word] += 1
            articlecounter += 1

    return allwords, articlewords, articletitles


def makematrix(allwords, articlewords):
    wordvec = []

    # Only take words that are common but not too common
    for w, c in allwords.items():
        if 3 < c < len(articlewords) * 0.6:
            wordvec.append(w)

    # Create the word matrix
    l1 = [[(word in article and article[word] or 1) for word in wordvec] for article in articlewords]
    return l1, wordvec


def showfeatures(w, h, articletitles, wordvec, out='features.txt'):
    outfile = open(out, 'w')
    pc, wc = np.shape(h)
    toppatterns = [[] for i in range(len(articletitles))]
    patternnames = []

    # Loop over all the features
    for i in range(pc):
        slist = []
        # Create a list of words and their weights
        for j in range(wc):
            slist.append((h[i, j], wordvec[j]))
        # Reverse sort the word list
        slist.sort()
        slist.reverse()

        # Print the first six elements
        n = [s[1] for s in slist[0:6]]
        outfile.write(str(n) + '\n')
        patternnames.append(n)

        # Create a list of articles for this feature
        flist = []
        for j in range(len(articletitles)):
            # Add the article with its weight
            flist.append((w[j, i], articletitles[j]))
            toppatterns[j].append((w[j, i], i, articletitles[j]))
        # Reverse sort the list
        flist.sort()
        flist.reverse()

        # Show the top 3 articles
        for f in flist[0:3]:
            outfile.write(str(f) + '\n')
        outfile.write('\n')

    outfile.close()
    # Return the pattern names for later use
    return toppatterns, patternnames


def showarticles(articletitles, toppatterns, patternnames, out='articles.txt'):
    outfile = open(out, 'w')

    # Loop over all the articles
    for j in range(len(articletitles)):
        outfile.write(articletitles[j] + '\n')

        # Get the top features for this article and reverse sort them
        toppatterns[j].sort()
        toppatterns[j].reverse()

        # Print the top three patterns
        for i in range(3):
            outfile.write(str(toppatterns[j][i][0]) + ' ' +
                          str(patternnames[toppatterns[j][i][1]]) + '\n')
        outfile.write('\n')

    outfile.close()
