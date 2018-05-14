import clusters
import newsfeatures
import docclass
import numpy as np

import nmf

feedlist = [
    'https://rsshub.app/apnews/topics/apf-topnews',
    'http://www.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'http://news.google.com/?output=rss',
    'http://www.foxnews.com/xmlfeed/rss/0,4313,0,00.rss',
    'http://rss.cnn.com/rss/edition_world.rss',
    'http://rss.cnn.com/rss/edition_us.rss'
]


def wordmatrixfeatures(article):
    return [wordsvector[word] for word in range(len(article)) if article[word] > 0]


allwords, articlewords, articletitles = newsfeatures.getarticlewords(feedlist)
wordmatrix, wordsvector = newsfeatures.makematrix(allwords, articlewords)

print("All words list [0:10]: {:s}".format(str(wordsvector[0:10])))
print("Articles[0]: {:s}".format(str(articletitles[0])))
print("Articles[1]: {:s}".format(str(articletitles[1])))
print("Articles[2]: {:s}".format(str(articletitles[2])))
print("Word list for {:s}: {:s}".format(articletitles[0], str(wordmatrix[0][0:20])))

print("Word Matrix features for article '{:s}': {:s}".format(articletitles[0], str(wordmatrixfeatures(wordmatrix[0]))))
# # Naive Bayes Classification
naivebayesclassifier = docclass.naivebayes(wordmatrixfeatures)
naivebayesclassifier.setdb('newtest.db')
# Train this as an 'swimming' story
naivebayesclassifier.train(wordmatrix[0], 'swimming')
# Train this as an 'football' story
naivebayesclassifier.train(wordmatrix[1], 'football')
# # How is this story classified?
print("Article with title '{:s}' classified as: '{:s}'".format(articletitles[1], naivebayesclassifier.classify(wordmatrix[1])))
print("Article with title '{:s}' classified as: '{:s}'".format(articletitles[2], naivebayesclassifier.classify(wordmatrix[2])))

# Hierarchical clustering
clust = clusters.hcluster(wordmatrix)
articletitlesUtf8 = [article.encode('utf-8') for article in articletitles]
clusters.drawdendrogram(clust, articletitlesUtf8, jpeg='news.jpg')

weights, features = nmf.factorize(np.matrix(wordmatrix), pc=20, iter=50)
topp, pn = newsfeatures.showfeatures(weights, features, articletitles, wordsvector)
print("Top patterns/features: {:s} and their names: {:s}".format(str(topp), str(pn)))
newsfeatures.showarticles(articletitles, topp, pn)
