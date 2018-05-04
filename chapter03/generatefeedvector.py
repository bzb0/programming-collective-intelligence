import re

import feedparser


# Returns title and dictionary of word counts for an RSS feed
def getWordCounts(url):
    # Parse the feed
    d = feedparser.parse(url)
    wc = {}

    if 'title' not in d.feed:
        return None

    # Loop over all the entries
    for e in d.entries:
        if 'summary' in e:
            summary = e.summary
        else:
            summary = e.description

        # Extract a list of words
        words = getWords(e.title + ' ' + summary)
        for word in words:
            wc.setdefault(word, 0)
            wc[word] += 1

    return d.feed.title, wc


def getWords(html):
    # Remove all the HTML tags
    txt = re.compile(r'<[^>]+>').sub('', html)

    # Split words by all non-alpha characters
    words = re.compile(r'[^A-Z^a-z]+').split(txt)

    # Convert to lowercase
    return [word.lower() for word in words if word != '']


feedCount = 0
apcount = {}  # number of blogs each word appeared in
wordCounts = {}
feedUrlFile = open('feedlist.txt', 'r')
for feedurl in feedUrlFile.readlines():
    result = getWordCounts(feedurl)
    if result is None:
        continue

    title, wc = result
    wordCounts[title] = wc

    for word, count in wc.items():
        apcount.setdefault(word, 0)
        if count > 1:
            apcount[word] += 1

    feedCount += 1

# Reducing the total number of words included in the file
wordlist = []
for word, blogCount in apcount.items():
    frac = float(blogCount) / feedCount
    if 0.1 < frac < 0.5:
        wordlist.append(word)

out = open('blogdata_out.txt', 'w')
out.write('Blog')
for word in wordlist:
    out.write('\t%s' % word)
out.write('\n')

for blog, wc in wordCounts.items():
    out.write(blog)
    for word in wordlist:
        if word in wc:
            out.write('\t%d' % wc[word])
        else:
            out.write('\t0')
    out.write('\n')
