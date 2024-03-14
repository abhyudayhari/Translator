import configparser
import torch
import torch.nn.functional as f
import matplotlib.pyplot as plt
import torch.nn as nn
import pandas as pd
import pickle
#from preprocessor import encode

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

#device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Differentiator(nn.Module):
  def __init__(self,n_embed,heads):
    super().__init__()     #n_embed is the embeddings after matching it with the embedding table and heads is the number of features
    self.key=nn.Linear(n_embed,heads,bias=False)
    self.query=nn.Linear(n_embed,heads,bias=False)
    self.value=nn.Linear(n_embed,heads,bias=False)
  def __call__(self,x):
    key=self.key(x)
    query=self.query(x)
    value=self.value(x)    # B,N,T, HEADS
    return [key,query,value]
class Multiplier(nn.Module):
  def __init__(self):    #batch, number,  block b,n,t
    super().__init__()

  def __call__(self,k,q,v,n_embed):
      #B,T,HEADS
    key,query=k,q
    wei=query@key.transpose(-2,-1)*(n_embed**-0.5)  #B,N,T,T
    return [wei,v]
class Masking(nn.Module):
  def __init__(self,block):
    super().__init__()
    self.register_buffer("tril",torch.tril(torch.ones(block,block)))
  def __call__(self,x):
    B,T,C=x.shape
    x=x.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

    return x
class Head(nn.Module):
  def __init__(self,n_embed,heads,block,dropout,mask=False):
    super().__init__()
    self.n_embed=n_embed
    self.mult=Multiplier()
    self.mask=Masking(block)
    self.maskistrue=mask
    self.dropout=nn.Dropout(dropout)
  def __call__(self,k,q,v):

    wei,v=self.mult(k,q,v,self.n_embed)
    if self.maskistrue:
      wei=self.mask(wei)

    wei=f.softmax(wei,dim=-1)

    wei=self.dropout(wei)
    out =wei @ v
    return out
class MultipleHead(nn.Module):
  def __init__(self,n_embed,heads,block,dropout,n_heads,mask=False):
    super().__init__()
    self.multihead=nn.ModuleList(Head(n_embed,heads,block,dropout,mask) for _ in range(n_heads))
  def __call__(self,k,q,v):
    return torch.cat([h(k,q,v) for h in self.multihead],dim=-1)

class FeedForward(nn.Module):
   def __init__(self,n_embed,factor,dropout):    #factor is the ratio of the no of new neurons to the original
      super().__init__()
      self.net=nn.Sequential(nn.Linear(n_embed,factor*n_embed),
                             nn.GELU(),
                             nn.Linear(factor*n_embed,n_embed),
                             nn.Dropout(dropout)
      )
   def __call__(self,x):
        out=self.net(x)
        return out

class EncoderBlock(nn.Module):
  def __init__(self,n_embed,factor,block,dropout,n_heads):
    super().__init__()
    self.lis=Differentiator(n_embed,n_embed//n_heads)
    self.multihead=MultipleHead(n_embed,n_embed//n_heads,block,dropout,n_heads,mask=False)
    self.ln1=nn.LayerNorm(n_embed)
    self.ffd=FeedForward(n_embed,factor,dropout)
    self.ln2=nn.LayerNorm(n_embed)
  def __call__(self,x):
    a=self.ln1(x)
    k,q,v=self.lis(a)[0],self.lis(a)[1],self.lis(a)[2]
    x=a+self.multihead(k,q,v)
    x=self.ffd(self.ln2(x))+x
    return x

class DecoderBlock(nn.Module):
  def __init__(self,n_embed,factor,block,dropout,n_heads):
    super().__init__()
    self.lis=Differentiator(n_embed,n_embed//n_heads)
    self.lis1=Differentiator(n_embed,n_embed//n_heads)
    self.lis2=Differentiator(n_embed,n_embed//n_heads)
    self.multihead=MultipleHead(n_embed,n_embed//n_heads,block,dropout,n_heads,mask=True)
    self.multihead1=MultipleHead(n_embed,n_embed//n_heads,block,dropout,n_heads,mask=False)
    self.ln1=nn.LayerNorm(n_embed)
    self.ffd=FeedForward(n_embed,factor,dropout)
    self.ln2=nn.LayerNorm(n_embed)
  def __call__(self,c):

    x,encodedata=c
    a=self.ln1(x)
    k,q,v=self.lis(a)[0],self.lis(a)[1],self.lis(a)[2]
    x=a+self.multihead(k,q,v)
    key,value=self.lis1(encodedata)[0],self.lis1(encodedata)[2]
    query=self.lis2(x)[1]
    x=x+self.multihead1(key,query,value)
    x=self.ffd(self.ln2(x))+x
    return [x,encodedata]
  
from preprocessor import encode
class Model(nn.Module):
  #self,n_embed,factor,block,dropout,n_heads
  def __init__(self,n_embed,factor,block,dropout,n_heads,n_layer,inputvocab_size,outputvocab_size,maxlen,device):
    super().__init__()
    self.device=device
    self.maxlen=maxlen
    self.block=block
    self.embed1=nn.Embedding(inputvocab_size,n_embed)
    #self.posembed1=positional_encoding(block,n_embed)
    #self.posembed=positional_encoding(block,n_embed)
    self.posembed1=nn.Embedding(block,n_embed)
    self.embed2=nn.Embedding(outputvocab_size,n_embed)
    self.posembed2=nn.Embedding(block,n_embed)
    self.linhead=nn.Linear(n_embed,outputvocab_size)
    self.enc=nn.Sequential(*[EncoderBlock(n_embed=n_embed,n_heads=n_heads,factor=factor,block=block,dropout=dropout) for _ in range(n_layer)])
    self.dec=nn.Sequential(*[DecoderBlock(n_embed=n_embed,n_heads=n_heads,factor=factor,block=block,dropout=dropout) for _ in range(n_layer)])
    self.ln=nn.LayerNorm(n_embed)
    self.ln1=nn.LayerNorm(n_embed)
    self.dropout=nn.Dropout(dropout)
    self.dropout1=nn.Dropout(dropout)
  def __call__(self,encin,decin,decout=None):
    B,T=encin.shape
    b,t=decin.shape

    enctokenembed=self.embed1(encin)  # b,t,cdevice
    encposembed=self.posembed1(torch.arange(T,device=self.device))
    xenc=enctokenembed+encposembed
    xenc=self.dropout(xenc)
    dectokenembed=self.embed2(decin)  # b,t,c
    decposembed=self.posembed2(torch.arange(t,device=self.device))

    xdec=dectokenembed+decposembed
    xdec=self.dropout1(xdec)
    encoutput=self.enc(xenc)
    encoutput=self.ln(encoutput)
    del enctokenembed,encposembed,xenc,dectokenembed,decposembed
    decoutput=self.dec([xdec,encoutput])
    decoutput=decoutput[0]
    decoutput=self.ln1(decoutput)

    logits=self.linhead(decoutput)  #b,t,vocab_size
    del decoutput
    b,t,c=logits.shape

    if decout==None:
      loss=None
    else:
      loss=f.cross_entropy(logits.reshape(b*t,-1),decout.reshape(-1))
    return loss,logits
  def generate(self,i,inputmerges,outputmerges):
    ix=torch.ones((1,1),dtype=torch.long,device=self.device)*1003
    a=torch.tensor([encode(text=i,type='input',inputmerges=inputmerges,outputmerges=outputmerges)+[1001]*(self.maxlen-len(encode(i,type='input',inputmerges=inputmerges,outputmerges=outputmerges)))])
    a=a.to(self.device)
    while ix[-1][-1]!=1002:
      ix_cond=ix[:,-self.block:]

      loss,logits=self(a,ix_cond)
      # print(logits.shape)
      logits=logits[:,-1,:]
      probs=f.softmax(logits,dim=1)
      #print(probs.shape)
      idx_next=torch.multinomial(probs,num_samples=1)
      #idx_next=idx_next.view(1,-1)
      ix=torch.cat((ix,idx_next),dim=1)

    return ix


#m.load_state_dict(torch.load("/content/drive/MyDrive/data_for_translator/model_val=0.9094.pt"))

