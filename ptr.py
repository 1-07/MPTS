import os
import torch

import torch.optim
import torch.nn.functional as F
from data import DataIter
from ptrmodel import STN

    
def trg_to_src(trg,s_num):
    b=torch.zeros((trg.shape[1],s_num-2), device=trg.device,dtype=torch.float32)
    a=trg.T-1
    for i,line in enumerate(a):
        b[i][line]=0.2
    return b.T


def train(model,iterator, optimizer,criterion, clip):
    model.train()
    
    epoch_loss = 0
    for i, batch in enumerate(iterator):
        (src,src_len) = batch.text         
        trg = batch.label         
        (ner,ner_len) = batch.ner
       

        optimizer.zero_grad()
        
        if hasattr(torch.cuda,'empty_cache'):
            torch.cuda.empty_cache()
        seq_len = src.shape[0]
        trgS=trg_to_src(trg,seq_len)
        output=model(src,src_len,ner,ner_len)      
        loss = criterion(output, trgS)

        with torch.no_grad():
            if i == 0:
                print("----------------------------------------------------------")
                print("    trgS:{}".format(trgS.T[0]))
                print("    output:{}".format(output.T[0]))
                print("    loss:{}".format(loss))
                print("----------------------------------------------------------")
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        if hasattr(torch.cuda,'empty_cache'):
            torch.cuda.empty_cache()

    return (epoch_loss / len(iterator)),grad_norm


  