import configparser
import pandas as pd
import pickle
import os.path
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
numofexamples=hyperparameters["numofexamples"]

def encode(text,inputmerges,outputmerges,type="input",):
   # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  if type=='input':
    merges=inputmerges
  else:
    merges=outputmerges
  while len(tokens) >= 2:
    stats = check_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

def decode(ids,type='input'):
  # given ids (list of integers), return Python string
  if type=='input':
    vocab=inputvocab
  else:
    vocab=outputvocab
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text
def check_stats(tokens):

  stats={}
  for i,b in zip(tokens,tokens[1:]):
    stats[(i,b)]=stats.get((i,b),0)+1
  return stats
def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids
# len(merge(a))
def createmerge(text,max_vocab):
  tokens=text.encode('utf-8')
  tokens=list(map(int,tokens))
  ids = list(tokens)
  merges={}
  num_merges=max_vocab-255

  for i in range(num_merges):
    stats = check_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    #print(f"merging {pair} into a new token {idx}")
    ids = merge(ids, pair, idx)
    merges[pair] = idx
  return merges

if __name__=="__main__":
    
    if os.path.isfile("my-dataset-train.csv") and os.path.isfile('my-dataset-validation.csv'):
       print("found data...")
    else:
       print("Downloading dataset......")
       from datasets import load_dataset
       raw_datasets = load_dataset("kaitchup/opus-English-to-French")

       for split, dataset in raw_datasets.items():
            dataset.to_csv(f"my-dataset-{split}.csv", index=None)
    print("reading dataset.....")
    train=pd.read_csv("my-dataset-train.csv")
    val=pd.read_csv('my-dataset-validation.csv')
    train[['english','french']]=train['text'].str.split('###>',expand=True)
    val[['english','french']]=val['text'].str.split('###>',expand=True)
    if numofexamples==None:
        traininput=train['english'].tolist()
        valinput=val['english'].tolist()
        trainoutput=train['french'].tolist()
        valoutput=val['french'].tolist()

    else: 
        traininput=train['english'][:numofexamples].tolist()
        valinput=val['english'][:numofexamples].tolist()
        trainoutput=train['french'][:numofexamples].tolist()
        valoutput=val['french'][:numofexamples].tolist()
    del train,val
    print("Dataset reading completed")

    print("creating input and output vocab and merge tables")
    inputtokenizertext=str(traininput[:hyperparameters['inputvocabtrainerdata']])
    outputtokenizertext=str(trainoutput[:hyperparameters["outputvocabtrainerdata"]])
    inputmerges=createmerge(inputtokenizertext,hyperparameters["max_vocab_size"])
    outputmerges=createmerge(outputtokenizertext,hyperparameters["max_vocab_size"])
    inputvocab = {idx: bytes([idx]) for idx in range(256)}
    outputvocab={idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in inputmerges.items():
        inputvocab[idx] = inputvocab[p0] + inputvocab[p1]

    for (p0, p1), idx in outputmerges.items():
        outputvocab[idx] = outputvocab[p0] + outputvocab[p1]

    inputvocab[1001]=b'|padding|'
    inputvocab[1002]=b'|eot|'
    inputvocab[1003]=b'|start|'
    outputvocab[1001]=b'|padding|'
    outputvocab[1002]=b'|eot|'
    outputvocab[1003]=b'|start|'
    inputvocab_size=len(inputvocab)
    outputvocab_size=len(outputvocab)

   
    print("Trimming.....")
    data=[]
    for i in ["train","val"]:
        if i =="train":
            inp=traininput
            out=trainoutput
        else:
            inp=valinput
            out=valoutput
        inp1,out1=[],[]
        for r in range(len(inp)):
            # print(r)
            # print(len(encode(inp[r])))
            if len(encode(text=inp[r],type='input',inputmerges=inputmerges,outputmerges=outputmerges))<=hyperparameters['maxlen'] and len(encode(text=out[r],type='output',inputmerges=inputmerges,outputmerges=outputmerges))<=hyperparameters['maxlen']:
                inp1+=[inp[r]]
                out1+=[out[r]]
        data+=[inp1,out1]
    print("Trimming completed")
    traininput1,trainoutput1,valinput1,valoutput1=data
    print("Encoding.......")
    totraininput=[[1003]+encode(text=i,type='input',inputmerges=inputmerges,outputmerges=outputmerges)+[1001]*(hyperparameters['maxlen']-len(encode(text=i,type='input',inputmerges=inputmerges,outputmerges=outputmerges)))+[1002] for i in traininput1]
    totrainoutput=[[1003]+encode(text=i,type='output',inputmerges=inputmerges,outputmerges=outputmerges)+[1001]*(hyperparameters['maxlen']-len(encode(text=i,type='output',inputmerges=inputmerges,outputmerges=outputmerges)))+[1002] for i in trainoutput1]
    tovalinput=[[1003]+encode(text=i,type='input',inputmerges=inputmerges,outputmerges=outputmerges)+[1001]*(hyperparameters['maxlen']-len(encode(text=i,type='input',inputmerges=inputmerges,outputmerges=outputmerges)))+[1002] for i in valinput1]
    tovaloutput=[[1003]+encode(text=i,type='output',inputmerges=inputmerges,outputmerges=outputmerges)+[1001]*(hyperparameters['maxlen']-len(encode(text=i,type='output',inputmerges=inputmerges,outputmerges=outputmerges)))+[1002] for i in valoutput1]
    a=[totraininput,totrainoutput,tovalinput,tovaloutput]
    print("Encoding completed,dumping.....")
    file=open("tokenized.dat","wb")
    pickle.dump(a,file)
    file.close()
    inputfile=open("inputfile.dat","wb")
    outputfile=open('outputfile.dat',"wb")
    inputfiles={'inputvocab':inputvocab,'inputmerges':inputmerges}
    outputfiles={'outputvocab':outputvocab,'outputmerges':outputmerges}
    pickle.dump(inputfiles,inputfile)
    pickle.dump(outputfiles,outputfile)
    inputfile.close()
    outputfile.close()

    print("Dumped all files to external dat files.....")
