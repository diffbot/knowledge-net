import itertools
import numpy as np
import networkx as nx
import vocab

def embedding(instance):
  def feature(word):
    # TODO try word dropout?
    return word.vector
  return feature

def bert_embedding(instance):
  def feature(word):
    return word._.bert_vector
  return feature

def bert_layers(instance):
  def feature(word):
    return word._.bert_layers
  return feature

IOB_TAGS = ['B','I','O']
TYPE_TAGS = ['', 'PERSON', 'NORP', 'FAC', 'ORG', 'GPE', 'LOC', 'PRODUCT', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE', 'DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']
pronoun_lists = {
  "PERSON": ["i", "you", "he", "she", "we", "they", "me", "him", "her", "us", "them",
            "who", "whom", "mine", "yours", "his", "hers", "ours", "theirs", "whose",
            "myself", "yourself", "himself", "herself", "ourselves", "themselves",
            "their", "our", "your", "my"],
  "ORG": ["it", "its", "we", "they", "us", "them", "ours", "theirs", "whose",
            "itself", "ourselves", "themselves",
            "where", "which", "their", "our"],
  "GPE": ["where", "there", "it"],
  "DATE": ["when", "whenever"],
  "": ["what", "within", "why"]
}
def ner_types(instance):
  def feature(word):
    larr = np.zeros(len(IOB_TAGS))
    larr[IOB_TAGS.index(word.ent_iob_)] = 1
    tarr = np.zeros(len(TYPE_TAGS))
    tarr[TYPE_TAGS.index(word.ent_type_)] = 1
    word_text = word.text.lower()
    for t, pronoun_list in pronoun_lists.items():
      if word_text in pronoun_list:
        tarr[TYPE_TAGS.index(t)] = 1
        larr[IOB_TAGS.index('B')] = 1
    return np.concatenate([larr, tarr])
  return feature

def coref_scores(instance):
  subject_entity = instance.subject_entity
  object_entity = instance.object_entity
  def feature(word):
    #subject
    if word in subject_entity:
      subject_coref_score = 5.0
    else:
      subject_coref_score = -5.0
      # for all mentions that contain this word and are in a cluster
      word_cluster_mentions = []
      for word_cluster in word._.coref_clusters:
        for mention in word_cluster.mentions:
          if word in mention:
            word_cluster_mentions.append(mention)
      for cluster_mention in word_cluster_mentions:
        if subject_entity in cluster_mention._.coref_scores:
          score = min(4.5, cluster_mention._.coref_scores[subject_entity])
          if score > subject_coref_score:
            subject_coref_score = score
    #object
    if word in object_entity:
      object_coref_score = 5.0
    else:
      object_coref_score = -5.0
      # for all mentions that contain this word and are in a cluster
      word_cluster_mentions = []
      for word_cluster in word._.coref_clusters:
        for mention in word_cluster.mentions:
          if word in mention:
            word_cluster_mentions.append(mention)
      for cluster_mention in word_cluster_mentions:
        if object_entity in cluster_mention._.coref_scores:
          score = min(4.5, cluster_mention._.coref_scores[object_entity])
          if score > object_coref_score:
            object_coref_score = score
    return [subject_coref_score/5.0, object_coref_score/5.0]
  return feature

def coref_flags(instance):
  subject_entity = instance.subject_entity
  object_entity = instance.object_entity
  subject_cluster = subject_entity._.coref_cluster if subject_entity._.coref_cluster is not None else [subject_entity]
  object_cluster = object_entity._.coref_cluster if object_entity._.coref_cluster is not None else [object_entity]
  subject_clusters_start_indices = list(map(lambda x: x.start, subject_cluster))
  subject_clusters_last_indices = list(map(lambda x: x.end-1, subject_cluster))
  object_clusters_start_indices = list(map(lambda x: x.start, object_cluster))
  object_clusters_last_indices = list(map(lambda x: x.end-1, object_cluster))
  def feature(word):
    return [
      # word in subject cluster
      1 if word in subject_entity or subject_cluster in word._.coref_clusters else 0,
      1 if (word in subject_entity and word.i == subject_entity.start ) or (subject_cluster in word._.coref_clusters and word.i in subject_clusters_start_indices) else 0,
      1 if (word in subject_entity and word.i == subject_entity.end-1 ) or (subject_cluster in word._.coref_clusters and word.i in subject_clusters_last_indices) else 0,
      # word in object cluster
      1 if word in object_entity or object_cluster in word._.coref_clusters else 0,
      1 if (word in object_entity and word.i == object_entity.start ) or (object_cluster in word._.coref_clusters and word.i in object_clusters_start_indices) else 0,
      1 if (word in object_entity and word.i == object_entity.end-1 ) or (object_cluster in word._.coref_clusters and word.i in object_clusters_last_indices) else 0,
    ]
  return feature

def dep_path(instance, coref=True):
  subject_entity = instance.subject_entity
  object_entity = instance.object_entity

  def edges_for_sentence(sentence):
    edges = []
    for token in sentence:
      for child in token.children:
        edges.append((token.i,child.i))
    return edges

  graph = nx.Graph()

  first_entity = subject_entity if subject_entity.start < object_entity.start else object_entity
  last_entity = object_entity if subject_entity.start < object_entity.start else subject_entity
  if subject_entity.sent == object_entity.sent:
    words = subject_entity.sent
    graph.add_nodes_from([ word.i for word in words])
    graph.add_edges_from(edges_for_sentence(subject_entity.sent))
  else:
    #print("sentences not equal")
    doc = subject_entity.doc
    first_entity = subject_entity if subject_entity.start < object_entity.start else object_entity
    last_entity = object_entity if subject_entity.start < object_entity.start else subject_entity
    words = doc[first_entity.sent.start:last_entity.sent.end]
    graph.add_nodes_from([ word.i for word in words])
    # dep graph
    graph.add_edges_from(edges_for_sentence(subject_entity.sent))
    graph.add_edges_from(edges_for_sentence(object_entity.sent))
    # add edge from root of both sentences
    graph.add_edge(subject_entity.sent.root.i,object_entity.sent.root.i)

  word_indices_in_shortest_path = set(nx.shortest_path(graph, source=subject_entity.root.i, target=object_entity.root.i))

  subject_cluster = subject_entity._.coref_cluster if subject_entity._.coref_cluster is not None else [subject_entity]
  object_cluster = object_entity._.coref_cluster if object_entity._.coref_cluster is not None else [object_entity]
  
  word_indices_in_subject = set()
  for word in subject_entity:
    word_indices_in_subject.add(word.i)
  word_indices_in_object = set()
  for word in object_entity:
    word_indices_in_object.add(word.i)

  word_indices_in_subject_cluster = set()
  for mention in subject_cluster:
    for word in mention:
      word_indices_in_subject_cluster.add(word.i)
  word_indices_in_object_cluster = set()
  for mention in object_cluster:
    for word in mention:
      word_indices_in_object_cluster.add(word.i)

  word_indices_in_clusters_shortest_path = set(word_indices_in_shortest_path)
  for subject_mention, object_mention in itertools.product(subject_cluster, object_cluster):
    if(subject_mention.root in words and object_mention.root in words):
      word_indices_in_clusters_shortest_path.update(nx.shortest_path(graph, source=subject_mention.root.i, target=object_mention.root.i))
  
  lengths = dict(nx.all_pairs_shortest_path_length(graph, cutoff=20))
  if coref:
    def feature(word):
      return [
        # how far is word from subject and closest subject cluster
        min([ lengths[word.i][spath_word] if word.i in lengths and spath_word in lengths[word.i] else 24 for spath_word in word_indices_in_subject ]),
        min([ lengths[word.i][spath_word] if word.i in lengths and spath_word in lengths[word.i] else 24 for spath_word in word_indices_in_subject_cluster ]),
        # how far is word from object and closest object cluster
        min([ lengths[word.i][spath_word] if word.i in lengths and spath_word in lengths[word.i] else 24 for spath_word in word_indices_in_object ]),
        min([ lengths[word.i][spath_word] if word.i in lengths and spath_word in lengths[word.i] else 24 for spath_word in word_indices_in_object_cluster ]),
        # how far is word from shortest path between subject and object
        min([ lengths[word.i][spath_word] if word.i in lengths and spath_word in lengths[word.i] else 24 for spath_word in word_indices_in_shortest_path ]),
        # how far is word from shortest path between subject clusters and object clusters
        min([ lengths[word.i][spath_word] if word.i in lengths and spath_word in lengths[word.i] else 24 for spath_word in word_indices_in_clusters_shortest_path ])
      ]
  else:
    def feature(word):
      return [
        # how far is word from subject
        min([ lengths[word.i][spath_word] if word.i in lengths and spath_word in lengths[word.i] else 24 for spath_word in word_indices_in_subject ]),
        # how far is word from object
        min([ lengths[word.i][spath_word] if word.i in lengths and spath_word in lengths[word.i] else 24 for spath_word in word_indices_in_object ]),
        # how far is word from shortest path between subject and object
        min([ lengths[word.i][spath_word] if word.i in lengths and spath_word in lengths[word.i] else 24 for spath_word in word_indices_in_shortest_path ]),
      ]

  return feature

def _position_to(mention):
  def feature(word):
    if word in mention:
      return [0]
    if word.i < mention.start:
      return [ mention.start - word.i ]
    return [ word.i - mention.end + 1 ]
  return feature

def subject_position(instance):
  return _position_to(instance.subject_entity)

def object_position(instance):
  return _position_to(instance.object_entity)

def cluster_positions(instance):
  subject_entity = instance.subject_entity
  object_entity = instance.object_entity
  subject_cluster = subject_entity._.coref_cluster if subject_entity._.coref_cluster is not None else [subject_entity]
  object_cluster = object_entity._.coref_cluster if object_entity._.coref_cluster is not None else [object_entity]

  word_indices_in_subject_cluster = set()
  for mention in subject_cluster:
    for word in mention:
      word_indices_in_subject_cluster.add(word.i)
  word_indices_in_object_cluster = set()
  for mention in object_cluster:
    for word in mention:
      word_indices_in_object_cluster.add(word.i)

  def feature(word):
    return [
      # how far away is the closest subject mention on the right
      min([cword_i - word.i for cword_i in word_indices_in_subject_cluster if cword_i >= word.i], default=24),
      # how far away is the closest subject mention on the left
      min([word.i - cword_i for cword_i in word_indices_in_subject_cluster if cword_i <= word.i], default=24),
      # how far away is the closest object mention on the right
      min([cword_i - word.i for cword_i in word_indices_in_object_cluster if cword_i >= word.i], default=24),
      # how far away is the closest object mention on the left
      min([word.i - cword_i for cword_i in word_indices_in_object_cluster if cword_i <= word.i], default=24),
    ]
  return feature
