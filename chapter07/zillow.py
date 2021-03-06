import xml.dom.minidom
import urllib3
import treepredict

zwskey = "X1-ZWz1chwxis15aj_9skq6"


def getaddressdata(address, city):
    escad = address.replace(' ', '+')

    # Construct the URL
    url = 'http://www.zillow.com/webservice/GetDeepSearchResults.htm?'
    url += 'zws-id=%s&address=%s&citystatezip=%s' % (zwskey, escad, city)
    http = urllib3.PoolManager()
    response = http.request('GET', url)

    # Parse resulting XML
    doc = xml.dom.minidom.parseString(response.data)
    code = doc.getElementsByTagName('code')[0].firstChild.data

    # Code 0 means success; otherwise, there was an error
    if code != '0':
        return None

    # Extract the info about this property
    try:
        zipcode = doc.getElementsByTagName('zipcode')[0].firstChild.data
        use = doc.getElementsByTagName('useCode')[0].firstChild.data
        year = doc.getElementsByTagName('yearBuilt')[0].firstChild.data
        bath = doc.getElementsByTagName('bathrooms')[0].firstChild.data
        bed = doc.getElementsByTagName('bedrooms')[0].firstChild.data
        rooms = doc.getElementsByTagName('totalRooms')[0].firstChild.data
        price = doc.getElementsByTagName('amount')[0].firstChild.data
    except:
        return None

    return (zipcode, use, int(year), float(bath), int(bed), int(rooms), price)


def getpricelist():
    l1 = []
    file = open('addresslist.txt', 'r')
    for line in file.readlines():
        data = getaddressdata(line.strip(), 'Cambridge,MA')
        l1.append(data)
    return l1


## Testing the script

housedata = getpricelist()
housetree = treepredict.buildtree(housedata, scoref=treepredict.variance)
treepredict.drawtree(housetree,'housetree.jpg')
