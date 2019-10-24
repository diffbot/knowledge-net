from collections import Counter
import os
import wikidata

path = 'vocab'

def _types_vocab():
  if not os.path.exists(os.path.join(path,'types.vocab')):
    return []
  with open(os.path.join(path,'types.vocab'), 'r') as f:
    return f.read().splitlines()

def _properties_vocab():
  if not os.path.exists(os.path.join(path,'properties.vocab')):
    return []
  with open(os.path.join(path,'properties.vocab'), 'r') as f:
    return f.read().splitlines()

types = _types_vocab()
properties = _properties_vocab()

class Vocab():
  """Outputs vocab files for features that need it"""

  def __init__(self):
    self.types = Counter()
    self.properties = Counter()

  def call(self, instance):
    self.types.update([ t for cand in instance.object_entity._.uri_candidates for t in cand["types"] ])
    self.types.update([ t for cand in instance.subject_entity._.uri_candidates for t in cand["types"] ])
    subject_list = [x["uri"] for x in instance.subject_entity._.uri_candidates]
    object_list = [x["uri"] for x in instance.object_entity._.uri_candidates]
    wikidataProperties = wikidata.get_properties(subject_list,object_list)
    self.properties.update([p for l in wikidataProperties.values() for p in l if not p in ""])

  def save(self):
    self.save_types()
    self.save_properties()

  def save_types(self):
    print(self.types.most_common())
    with open(os.path.join(path,'types.vocab'), 'w') as file:
      for e,c in self.types.most_common()[:100]:
        file.write(e)
        file.write('\n')
    # reload
    types = _types_vocab()

  def save_properties(self):
    print(self.properties.most_common())
    with open(os.path.join(path,'properties.vocab'), 'w') as file:
      for e,c in self.properties.most_common():
        if c < 5:
          break
        file.write(e)
        file.write('\n')
    # reload
    properties = _properties_vocab()
