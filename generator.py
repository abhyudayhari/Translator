import configparser
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import pickle
from tqdm import tqdm
import pylab as pl
from IPython import display
import sys
from preprocessor import decode
def read_config(filename):
    config = configparser.ConfigParser()
    config.read(filename)
    return config

config_file = 'config.ini'
config = read_config(config_file)
hyperparameters={}
options = config.options("Hyperparameters")
for option in options:
    value = config.get("Hyperparameters", option)
    hyperparameters[option] = eval(value)
if hyperparameters["device"]==None:
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
elif hyperparameters["device"]==0:
   device=torch.device('cpu')
elif hyperparameters["device"]==1:
   device=torch.device('cuda')
else:
   print("Enter a valid device")
   exit(0)

inputfile=open("inputfile.dat","rb")
outputfile=open('outputfile.dat',"rb")
inputfiles=pickle.load(inputfile)
outputfiles=pickle.load(outputfile)
inputvocab=inputfiles["inputvocab"]
inputmerges=inputfiles["inputmerges"]
outputvocab=outputfiles["outputvocab"]
outputmerges=outputfiles["outputmerges"]
from model import Model
m=Model(n_embed=hyperparameters['n_embed'],factor=hyperparameters['factor'],block=hyperparameters['block'],n_heads=hyperparameters["n_heads1"],n_layer=hyperparameters["n_layer"],dropout=hyperparameters["dropout"],inputvocab_size=len(inputvocab),outputvocab_size=len(outputvocab),maxlen=hyperparameters["maxlen"],device=device)

def decode(ids,type='input'):
  # given ids (list of integers), return Python string
  if type=='input':
    vocab=inputvocab
  else:
    vocab=outputvocab
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text
if len(sys.argv)==1:
    print("Please enter a model file")
    exit(0)

dir=sys.argv[1]
m.load_state_dict(torch.load(dir))
m.to(device)

for _ in range(int(input("Enter the number of translations to be made"))):
    inputtext=input("Enter the input text!!!\n")
    print("Translating....\n")
    raw=m.generate(i=inputtext,inputmerges=inputmerges,outputmerges=outputmerges)[-1].tolist()
    raw=[i for i in raw if i not in [1002,1001,1003]]
    decode(raw)
    print("-----------------------------------------------------------------------------------\n")