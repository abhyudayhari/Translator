
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
from model import Model

file=open("tokenized.dat","rb")
a=pickle.load(file)
totraininput,totrainoutput,tovalinput,tovaloutput=a
file.close()
inputfile=open("inputfile.dat","rb")
outputfile=open('outputfile.dat',"rb")
inputfiles=pickle.load(inputfile)
outputfiles=pickle.load(outputfile)
inputvocab=inputfiles["inputvocab"]
inputmerges=inputfiles["inputmerges"]
outputvocab=outputfiles["outputvocab"]
outputmerges=outputfiles["outputmerges"]
m=Model(n_embed=hyperparameters['n_embed'],factor=hyperparameters['factor'],block=hyperparameters['block'],n_heads=hyperparameters["n_heads1"],n_layer=hyperparameters["n_layer"],dropout=hyperparameters["dropout"],inputvocab_size=len(inputvocab),outputvocab_size=len(outputvocab),maxlen=hyperparameters["maxlen"],device=device)
class lr_scheduler():
    def __init__(self,n_embed,warmup_steps=4000):
        self.n_embed=torch.tensor(n_embed,dtype=torch.float64)
        self.warmup_steps=torch.tensor(warmup_steps,dtype=torch.float64)
    def __call__(self,step):
        step=torch.tensor(step,dtype=torch.float64)
        #arg1=torch.sqrt(step)
        #arg2=step * (self.warmup_steps ** -1.5)

        return self.n_embed**-0.5*torch.min(step**-0.5,step*self.warmup_steps**-1.5)

lrs=lr_scheduler(hyperparameters['n_embed'],warmup_steps=hyperparameters["warmup_steps"])

def batches(split,batch,block):
  data=[totraininput,totrainoutput] if split=="train" else [tovalinput,tovaloutput]
  ix=torch.randint(len(data[0]),(batch,))
  datain,dataoutput=data[0],data[1]
  din=[datain[i] for i in ix]
  dout=[dataoutput[i] for i in ix]
  xin=torch.stack([torch.tensor(j[1:block]) for j in din])
  xout=torch.stack([torch.tensor(j[2:block+1]) for j in din])
  yin=torch.stack([torch.tensor(j[:block]) for j in dout])
  yout=torch.stack([torch.tensor(j[1:block+1]) for j in dout])
  xin,xout,yin,yout=xin.to(device),xout.to(device),yin.to(device),yout.to(device)
  return xin,xout,yin,yout

@torch.no_grad()
def estimate_loss(batch,block,eval_iters,model):
  out={}
  model.eval()
  print("estimating loss using functino")
  for split in ['train','val']:

    losses=torch.zeros(eval_iters)
    for k in tqdm(range(eval_iters)):
      ein,eout,din,dout=batches(split,batch,block)
      loss,logits=model(encin=ein,decin=din,decout=dout)
      losses[k] =loss.item()
    out[split]=losses.mean()
    print(out[split]," ",split)
  print(out)
  model.train()
  return out


def training(lossl=None):
  if lossl==None:
    lossl=[]
  else:
    lossl=lossl
  print("running on ",device)
  for steps in tqdm(range(hyperparameters["iters"])):
    ein,eout,din,dout=batches("train",batch=hyperparameters["batch"],block=hyperparameters["block"])
    lr=lrs(steps)
    optimizer = torch.optim.AdamW(m.parameters(), lr=lr,betas=(hyperparameters["beta1"],hyperparameters["beta2"]),eps=hyperparameters["eps"])
    loss,logits=m(encin=ein,decin=din,decout=dout)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    lossl+=[loss.item()]
    # if steps%eval_iters==0 or steps==iters-1:
    #    estimate_loss(batch,block,eval_iters,m)
    if steps%5000==0:
      torch.save(m.state_dict(), f"model{steps}.pt")

    if steps%200==0 or steps==hyperparameters["iters"]-1:

      pl.plot(range(len(lossl)),lossl,color="red")
      pl.axhline(y=1, color='b', linestyle='--')
      display.clear_output(wait=True)
      display.display(pl.gcf())
    
  torch.save(m.state_dict(), f"model{steps}.pt")
  return lossl
losses=[]
m.to(device)
losses=training(lossl=losses)