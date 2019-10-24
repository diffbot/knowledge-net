import requests
import os
from sqlitedict import SqliteDict
import time
import urllib

if not os.path.exists("./tmp"):
  os.makedirs("./tmp")
db = SqliteDict(os.path.join('./tmp','properties_subject.db'), autocommit=True)


from SPARQLWrapper import SPARQLWrapper, JSON


ENDPOINT = 'https://query.wikidata.org/sparql'

def get_query_from_subjects_only(subject):
  query = """
  SELECT ?s ?relation ?o
  WHERE {
    {BIND(wd:%s AS ?s). ?s ?relation ?o}
  }""" % (subject)
  return query

def is_valid_or_convert(uri):
  if uri == "":
    return None
  if uri[-1] == "/":
    uri = uri[:-1]
  uri = uri.split("/")[-1]
  if uri[0] == "Q":
    return uri
  else:
    return None

def request(subject, set_objects):
  try:
    triples = []
    wikidata_response = db.get(subject, None)
    if not wikidata_response is None:
      for t in wikidata_response.split("\n"):
        subject = t.split("\t")[0]
        property = t.split("\t")[1]
        object = t.split("\t")[2]
        if object in set_objects:
          triples.append((subject, property, object))
    else:
      sparql = SPARQLWrapper(ENDPOINT)
      query = get_query_from_subjects_only(subject)
      sparql.setQuery(query)
      sparql.setReturnFormat(JSON)
      data = sparql.query().convert()
      triples_2_store = ""
      for p in data["results"]["bindings"]:
        object = p["o"]["value"].split("/")[-1]
        subject = p["s"]["value"].split("/")[-1]
        property = p["relation"]["value"].split("/")[-1]
        triples_2_store = triples_2_store + subject + "\t" + property + "\t" + object + "\n"
        if object in set_objects:
          triples.append((subject, property, object))
      if triples_2_store != "":
        db[subject] = triples_2_store.rstrip("\n")
    return triples
  except urllib.error.HTTPError:
    time.sleep(1)
    return request(subject, set_objects)

def get_properties(list_first_uri, list_second_uri):
  list_first_uri_clean = [is_valid_or_convert(x) for x in list_first_uri if is_valid_or_convert(x) != None]
  list_second_uri_clean = [is_valid_or_convert(x) for x in list_second_uri if is_valid_or_convert(x) != None]
  propertiesBySubObj = {}
  cont = 0
  for subject in list_first_uri_clean:
    cont+=1
    triples = request(subject, set(list_second_uri_clean))
    for t in triples:
      entry = str(t[0]) + "-" + str(t[2])
      if not entry in propertiesBySubObj:
        propertiesBySubObj[entry] = set()
      propertiesBySubObj[entry].add(t[1])
  return propertiesBySubObj


if __name__ == "__main__":
  print(get_properties(["Q76", "Q242951"], ["Q1860"]))