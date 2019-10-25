#MODEL = "simple_pipeline"
#MODEL = "pipeline_without_global"
#MODEL = "best_pipeline"
#MODEL = "ours"
MODEL = "bert"

NUMBER_URI_CANDIDATES = 1 if MODEL == "ours" else 1
NUMBER_URI_CANDIDATES_TO_CONSIDER = 1
URI_THRESHOLD = 0.0
SOFT_COREF_CANDIDATES = MODEL == "ours" or MODEL == "bert"
MULTITASK = True
CANDIDATE_RECALL = False

USE_ENTITY_LINKER = True
USE_BERT = MODEL == "bert"

MODELS_DIR = "models"

KNOWLEDGE_NET_DIR = "../../"
