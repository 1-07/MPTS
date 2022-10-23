import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LA



class STN(nn.Module):
   
    def __init__(self, vocabulary_size,ner_vocab, emb_size,hidden_size, n_layers, dropout):
        super(STN, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.n_layers=n_layers
        self.ner_size=ner_vocab


        self.embedding = nn.Embedding(vocabulary_size,emb_size)
        self.gru = nn.GRU(emb_size, hidden_size, n_layers, dropout=(dropout if n_layers != 1 else 0), bidirectional=True)
        self.ngru = nn.GRU(16, 32, n_layers, dropout=(dropout if n_layers != 1 else 0), bidirectional=True)
        self.dropout = nn.Dropout(dropout)

        self.w=nn.Parameter(torch.Tensor(hidden_size*2, 1))
        self.nw=nn.Parameter(torch.Tensor(32*2, 1))
        self.act1=nn.Tanh()
        self.act2=nn.Tanh()
      
        self.embedding_ner=nn.Embedding(ner_vocab,16)
        self.cos=torch.nn.PairwiseDistance()
        nn.init.uniform_(self.w, -0.1, 0.1)
        nn.init.uniform_(self.nw, -0.1, 0.1)
        
    

    def forward(self, input, input_len,ner,ner_len,h_0=None):

        weight = []
        fx=self.cal_fx(input,input_len,ner,ner_len,h_0)
        for k in range(1,input.shape[0]-1):
            new_src,new_src_len,new_ner,new_ner_len=self.process_src(input,input_len,ner,ner_len,k)   
            fxk=self.cal_fx(new_src,new_src_len,new_ner,new_ner_len,h_0)    
            w=self.cos(fxk,fx)           
            weight.append(w)       
        f_output = torch.stack(weight)
        f_output = F.softmax(f_output,dim=0)     
         
        return f_output

    def cal_fx(self,src,src_len,ner,ner_len,h0):
        embedded=self.dropout(self.embedding(src))
        ner_emb=self.dropout(self.embedding_ner(ner))    

        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len.cpu())
        output, h_n = self.gru(packed, h0)
        output, output_lens = torch.nn.utils.rnn.pad_packed_sequence(output)

        npacked = nn.utils.rnn.pack_padded_sequence(ner_emb, ner_len.cpu())
        noutput, nh_n = self.ngru(npacked, None)
        noutput, noutput_lens = torch.nn.utils.rnn.pad_packed_sequence(noutput)
        output=output.permute(1,0,2)
        noutput=noutput.permute(1,0,2)
        
        m=self.act1(output)
        n=self.act1(noutput)
        alpha=F.softmax(m.matmul(self.w),dim=1)
        nalpha=F.softmax(n.matmul(self.nw),dim=1)
        r=output*alpha
        nr=r*nalpha
        nr=torch.sum(nr,dim=1)
        fx=self.act2(nr)
        return fx


    def process_src(self,src,src_len,ner,ner_len,k):
        new_src=src.T.clone()
        new_src_len = src_len.clone()
        new_ner=ner.T.clone()
        new_ner_len = ner_len.clone()
        len=src.shape[0]
        for i in range(src.shape[1]):
            if k < (src_len[i].item()-1):
                new_src[i][k:len-1]=src.T[i][k+1:len]
                new_src[i][len-1] = 1
                new_src_len[i] = new_src_len[i]-1
                new_ner[i][k:len-1]=ner.T[i][k+1:len]
                new_ner[i][len-1] = 1
                new_ner_len[i] = new_ner_len[i]-1
        return new_src.T,new_src_len,new_ner.T,new_ner_len


