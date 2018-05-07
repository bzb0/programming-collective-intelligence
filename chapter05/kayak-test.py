import kayak

sid = kayak.getkayaksession()
searchid = kayak.flightsearch(sid, 'BOS', 'LGA', '11/17/2006')
f = kayak.flightsearchresults(sid, searchid)
