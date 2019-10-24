from os import listdir
import os
import subprocess
import shutil
import sys
import tensorflow as tf
import pathlib

path = 'data'
config_name = 'data.yml'
with open(config_name, 'r') as file:
    config = file.read()

run_dir = 'run'

from config import MODELS_DIR
model_dir = MODELS_DIR
pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

# train models given as args (seperated by commas) or all predicates if none given
predicates = sys.argv[1].split(",") if len(sys.argv) > 1 else listdir(path)

def train_text(predicate_id):
  # copy predicate specific config
  data_dir = os.path.join(path, predicate_id)
  replace = {'data_dir':data_dir}
  pred_config = config.format(**replace)
  with open(os.path.join(data_dir,config_name), 'w') as f:
    f.write(pred_config)
  # train model
  should_stop = False
  try:
    ecode = subprocess.run(['onmt-main', 'train_and_eval', '--model', 'lstm.py', '--config', os.path.join(os.path.join(data_dir,config_name))])
    print(ecode)
  except KeyboardInterrupt:
    should_stop = True
  return should_stop
  
for predicate_id in predicates:
  for _ in range(1): # try again if training fails
    print(predicate_id)
    # delete run folder if exists
    if os.path.exists(run_dir) and os.path.isdir(run_dir):
      shutil.rmtree(run_dir)
    
    should_stop = train_text(predicate_id)

    # Move best model
    best_dir = os.path.join(run_dir,'export','best')
    if not os.path.exists(best_dir):
      print("NO MODEL FOR",predicate_id)
      if should_stop:
        break
      continue
    model_paths = [os.path.join(best_dir, basename) for basename in listdir(best_dir)]
    latest_file = max(model_paths, key=os.path.getctime)
    model_save_location = os.path.join(model_dir, predicate_id)
    if os.path.exists(model_save_location) and os.path.isdir(model_save_location):
      shutil.rmtree(model_save_location)
    shutil.move(latest_file, model_save_location)
    shutil.copyfile(os.path.join(path, predicate_id, "header.txt") ,os.path.join(model_save_location, "header.txt"))
    print("Saved model for", predicate_id)
    break
  if should_stop:
    break
  