import requests
from requests.adapters import HTTPAdapter
import os
from sqlitedict import SqliteDict
import hashlib
from spacy.tokens import Span
import json
import time
import diffbot_nlapi
import logging
import pathlib

from config import MODEL, NUMBER_URI_CANDIDATES, SOFT_COREF_CANDIDATES

# el_candidate has types, uri, score
Span.set_extension("el_candidates", default=[])
Span.set_extension("uri_candidates", default=[])

pathlib.Path('tmp').mkdir(parents=True, exist_ok=True)
db = SqliteDict(os.path.join('tmp','el.db'), autocommit=True)

configuration = diffbot_nlapi.Configuration()
api_instance = diffbot_nlapi.NaturalLanguageApi(diffbot_nlapi.ApiClient(configuration))

def _get_uri_candidates_from_mention_with_score(mention, score):
  return [ { 'types': elc["types"], 'uri': elc["uri"], 'score': (2*score)+elc["score"], 'coref_score':score, 'el_score':elc["score"]} for elc in mention._.el_candidates if SOFT_COREF_CANDIDATES or elc["score"] > 0.5 ]
def _coref_probability_from_score(score):
  # coref scores are not between 0 and 1, normalize
  return (max(-5.0,min(4.5,score)) + 5.0) / 10.0
def _get_uri_candidates(mention):
  ret = _get_uri_candidates_from_mention_with_score(mention, 1.0)
  # Look at all coref clusters and get el_candidates from each
  # score uri_candidates based on coref and and entitylinker score
  if MODEL == "best_pipeline" or MODEL == "pipeline_without_global":
    if len(ret) == 0 and mention._.coref_cluster:
      ret = _get_uri_candidates_from_mention_with_score(mention._.coref_cluster.main, 1.0)
  if SOFT_COREF_CANDIDATES:
    if mention._.coref_scores: #TODO we might want to look at overlapping mentions here?
      for coref_mention, score in mention._.coref_scores.items(): 
        normalized_score = _coref_probability_from_score(score)
        ret.extend(_get_uri_candidates_from_mention_with_score(coref_mention, normalized_score))
        for coref_mention_inner, score_inner in coref_mention._.coref_scores.items(): 
          ret.extend(_get_uri_candidates_from_mention_with_score(coref_mention_inner, normalized_score * _coref_probability_from_score(score_inner)))

    # sort and keep top n
    ret.sort(key=lambda x:x['score'], reverse=True)
  return ret[:NUMBER_URI_CANDIDATES]

def link(doc, mentions):
  offsets = []
  mentionDict = dict()
  for mention in mentions:
    if mention._.is_pronoun:
      continue
    offsets.append((mention.start_char, mention.end_char))
    mention_id = str(mention.start_char) + "-" + str(mention.end_char)
    mentionDict[mention_id] = mention
  mentions_str = str(sorted([ (ent.start_char, ent.end_char) for ent in mentions ]))
  cache_key = hashlib.md5((doc.text + mentions_str).encode()).hexdigest()
  el_response = db.get(cache_key, None)
  if el_response is None:
    el_response = _link(doc, offsets)
    db[cache_key] = el_response
  if 'mentions' not in el_response:
    logging.warning("No mentions returned for %s", doc.text.replace('\n', '.'))
    return
  for mention in el_response['mentions']:
    el_candidates = []
    if "entityCandidates" in mention:
      for scored_candidate in mention['entityCandidates']:
        uri = next((u for u in scored_candidate['allUris'] if "wikidata.org" in u), None) if 'allUris' in scored_candidate else None
        types = list(map(lambda x: x["name"], scored_candidate['allTypes'])) if 'allTypes' in scored_candidate else []
        if uri:
          el_candidates.append({'types': types, 'uri': uri, 'score': scored_candidate['confidence']})

    chunk_id = str(mention['beginOffset']) + "-" + str(mention['endOffset'])
    if chunk_id in mentionDict:
      mentionDict[chunk_id]._.el_candidates = el_candidates
  for mention in mentions:
    mention._.uri_candidates = _get_uri_candidates(mention)

def _link(doc, offsets):
  if 'DIFFBOT_TOKEN' not in os.environ:
    raise Exception("Must define environment variable DIFFBOT_TOKEN")
  DIFFBOT_TOKEN = os.environ['DIFFBOT_TOKEN']
  
  documents = [ 
    diffbot_nlapi.Document(content=doc.text.replace('\n', '.'),
    mentions=[diffbot_nlapi.Span(begin_offset=b, end_offset=e) for (b,e) in offsets]) 
  ]

  api_response = api_instance.v1_post(documents, DIFFBOT_TOKEN, fields=["mentions"])
  return api_response[0]


