#!/usr/bin/env python
# coding: utf-8

# <a 
# href="https://colab.research.google.com/github/wingated/cs474_labs_f2019/blob/master/DL_Lab7.ipynb"
#   target="_parent">
#   <img
#     src="https://colab.research.google.com/assets/colab-badge.svg"
#     alt="Open In Colab"/>
# </a>

# # Music Transformer
# ### Description:
# This notebook contains the architecture for a decoder-only transformer. It can be used to generate a long sequence of any type of token, but in this case, it was used for music notes. Its results can be heard here: https://soundcloud.com/joshbtay/sets/transformer-results. In a previous iteration, I attempted to use an LSTM to generate music. Its results can be heard here: https://soundcloud.com/joshbtay/sets/lstm-generative-results

# #### Based on "Attention is All You Need" (https://arxiv.org/abs/1706.03762) and "The Annotated Transformer" (https://nlp.seas.harvard.edu/2018/04/03/attention.html), altered to use decoder-only structure
# 
# 

# ## Setup
# 

# In[4]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context="talk")
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Model Helpers
# 

# In[3]:


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# ## Decoder
# 
# The decoder is also composed of a stack of $N=6$ identical layers.  
# 

# In[5]:


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, tgt_mask):
        for layer in self.layers:
            x = layer(x, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, tgt_mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        return self.sublayer[2](x, self.feed_forward)
    
def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


# In[6]:


def attention(query, key, value, mask):
    scale = np.sqrt(key.shape[2])
    scores = torch.bmm(query, key.permute(0, 2, 1)) / scale
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    output = torch.bmm(F.softmax(scores, dim = 2), value)
    return output


# In[7]:


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.dropout = dropout
        head = nn.Linear(d_model, d_model//h).cuda()
        self.attention_heads = nn.ModuleList([clone(head, 3) for _ in range(h)])
        self.w_output: nn.Linear = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask):
        output = []
        for head in self.attention_heads:
            q = head[0](query)
            k = head[1](key)
            v = head[2](value)
            output.append(attention(q, k, v, mask))
        outputs_concat = torch.cat(output, 2)
        multi_head = self.w_output(outputs_concat)
        return multi_head


# In[8]:


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = 1 / (10000 ** (torch.arange(0., d_model, 2) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)


# ## Full Model

# In[13]:


class TransformerModel(nn.Module):
    def __init__(self, tgt_vocab, N=6, d_model=256, d_ff=1024, h=8, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        c = copy.deepcopy
        
        self.decoder = Decoder(DecoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
        self.generator = Generator(d_model, tgt_vocab)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, tgt, tgt_mask):
        return self.decode(tgt, tgt_mask)
    
    def decode(self, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), tgt_mask)


# # Training
# 

# ## Batches and Masking

# In[14]:


class Batch:
    def __init__(self, trg=None, pad=0):
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask
    
    
global max_tgt_in_batch
def batch_size_fn(new, count, sofar):
    global max_tgt_in_batch
    if count == 1:
        max_tgt_in_batch = 0
    max_tgt_in_batch = max(max_tgt_in_batch,  len(new.trg) + 2)
    tgt_elements = count * max_tgt_in_batch
    return tgt_elements


# In[15]:


class LabelSmoothing(nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, Variable(true_dist, requires_grad=False))


# ## Data Loading
# 

# In[16]:


from torchtext.legacy import data, datasets
import torchtext

def tokenize(text):
    return text.split()

BOS_WORD = '<s>'
EOS_WORD = '</s>'
BLANK_WORD = "<blank>"
TGT = data.Field(tokenize=tokenize, init_token = BOS_WORD, eos_token = EOS_WORD, pad_token=BLANK_WORD)

print("Loading Dataset")
music_text = open('./data.txt', 'r')
music_lines = list(music_text)

fields = [("trg", TGT)]
examples = [torchtext.legacy.data.Example.fromlist([(music_lines[i])], fields) for i in range(len(music_lines))]

MAX_LEN = 1500
train, val = torchtext.legacy.data.Dataset(examples, fields=fields, filter_pred=lambda x: 
        len(vars(x)['trg']) <= MAX_LEN).split()

MIN_FREQ = 1
TGT.build_vocab(train.trg, min_freq=MIN_FREQ)


# ## Training Code

# In[17]:


class LossFunction:
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.zero_grad()
        return loss.data * norm

class DataIterator(data.Iterator):
    def create_batches(self):
        if self.train:
            def pool(d, random_shuffler):
                for p in data.batch(d, self.batch_size * 100):
                    p_batch = data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b
            self.batches = pool(self.data(), self.random_shuffler)
            
        else:
            self.batches = []
            for b in data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))

def rebatch(pad_idx, batch):
    trg = batch.trg.transpose(0, 1).cuda()
    return Batch(trg, pad_idx)

    
def run_epoch(data_iter, model, loss_compute, startt):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        if time.time() - startt > timeout:
          break
        out = model.forward(batch.trg, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f Time elapsed: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed, time.time() - startt))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
    


# ##Train

# In[8]:


import gc
gc.collect()
timeout = 14400
pad_idx = TGT.vocab.stoi["<blank>"]
model = TransformerModel(len(TGT.vocab), N=8).cuda()
n_epochs = 10000
device = torch.device('cuda')


# In[56]:


import gc
gc.collect()
timeout = 14400
pad_idx = TGT.vocab.stoi["<blank>"]
model = TransformerModel(len(TGT.vocab), N=8).cuda()
n_epochs = 10000
device = torch.device('cuda')

def scope():
    criterion = LabelSmoothing(size=len(TGT.vocab), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 1000
    train_iter = DataIterator(train, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.trg)),
batch_size_fn=batch_size_fn, train=True)
    valid_iter = DataIterator(val, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)

    model_opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    start = time.time()
    for epoch in range(n_epochs):
        if time.time() - start > timeout:
          break
        model.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter), 
                  model, 
                  LossFunction(model.generator, criterion, model_opt), start)
        model.eval()
scope()


# ## Translate

# In[ ]:


import random
def greedy_decode(model, src, max_len, start_symbol, start_seq):
    k=20
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)
    for sym in start_seq:
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(sym)], dim=1)
    for i in range(max_len-1):
        out = model.decode(Variable(ys), Variable(subsequent_mask(ys.size(1)).type_as(src.data)))
        prob = model.generator(out[:, -1])
        dist, indices = prob[0].sort()
        norm = []
        dist = dist[-k:]
        indices = indices[-k:]
        mi = min(dist)
        ma = max(dist)
        s = 0
        for j in dist:
          norm.append((j - mi)/(ma-mi))
          s+= norm[-1]
        p = random.random() ** 2 * s
        #p = random.uniform(0, s)
        s = 0
        for j,q in enumerate(norm):
          s += q
          if s >= p:
            next_word = indices[j]
            break
        #_, next_word = torch.max(prob, dim = 1)
        #next_word = next_word.data[0]
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
    return ys

BATCH_SIZE = 1000
n_train_iters = len(train) / BATCH_SIZE
valid_iter = DataIterator(val, batch_size=BATCH_SIZE, device=device,
                        repeat=False, sort_key=lambda x: (len(x.trg)),
                        batch_size_fn=batch_size_fn, train=False)
    
for i, batch in enumerate(valid_iter):
    start_seq = []
    for j in range(1, 5):
       start_seq.append(batch.trg.data[j, 0])
    src = batch.trg.transpose(0, 1)[:1].cuda()
    out = greedy_decode(model, src, max_len=200, start_symbol=TGT.vocab.stoi["<s>"], start_seq=start_seq)
    print("Translation:", end="\t")
    f = open(f"Outmidis/{i:04}.txt", "w")
    for j in range(1, out.size(1)):
        sym = TGT.vocab.itos[out[0, j]]
        if sym == "</s>": break
        print(sym, end =" ")
        f.write(sym+ " ")
    f.close()
    print()
    print("Target:\t", end="\t")
    for j in range(1, batch.trg.size(0)):
        sym = TGT.vocab.itos[batch.trg.data[j, 0]]
        if sym == "</s>": break
        print(sym, end =" ")
    print()
    print()
    
    if i > 1000 and i<1100:
        break

