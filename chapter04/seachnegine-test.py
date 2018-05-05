from chapter04 import searchengine, nn

neuralnet = nn.searchnet('nn.db')
searcher = searchengine.searcher('searchindex.db')

while True:
    query = input('Enter search string: ')
    wordids, urls = searcher.query(query)

    selectedurlid = input("Enter selected URL: ")
    neuralnet.trainquery(wordids, urls, selectedurlid)
