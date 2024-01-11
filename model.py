import torch
import torch.nn as nn

class Embedding_inverted(nn.Module):
    def __init__(self, T, D):
        super(Embedding_inverted, self).__init__()
        self.emb = nn.Linear(T, D)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = x.permute(0,2,1)
        x_emb = self.emb(x)
        return self.dropout(x_emb)

class FeedForward(nn.Module):
    def __init__(self, D):
        super(FeedForward, self).__init__()
        self.feed_forward = nn.Sequential(
          nn.Conv1d(in_channels=D, out_channels=D*4, kernel_size=1),
          nn.GELU(),
          nn.Dropout(0.1),
          nn.Conv1d(in_channels=D*4, out_channels=D, kernel_size=1)
        )
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        #print("avant : ", x.shape)
        x = x.permute(0,2,1)
        x = self.feed_forward(x)
        #print("pendant : ", x.shape)
        x = x.permute(0,2,1)
        #print("apres : ", x.shape)
        x = self.drop(x)
        return x

    
class Attention(nn.Module):
    def __init__(self, D, proj_dim, nb_head=8):
        super(Attention, self).__init__()
        self.query_projection = nn.Linear(D, proj_dim)
        self.key_projection = nn.Linear(D, proj_dim)
        self.value_projection = nn.Linear(D, proj_dim)
        self.out_projection = nn.Linear(proj_dim, D)
        self.H = nb_head
        self.dropout = nn.Dropout(0.1)

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.H

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = 1. / torch.sqrt(torch.tensor(E))

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        out = V.reshape(B,L,-1)

        return self.out_projection(out)
        

class TrmBlock(nn.Module):
    def __init__(self, N, D, proj_dim):
      super(TrmBlock, self).__init__()

      self.multivariate_attention= Attention(D, proj_dim)
      #self.multivariate_attention = nn.MultiheadAttention(D, num_heads=8)
      self.layer_norm1 = nn.LayerNorm(D)
      self.feed_forward = FeedForward(D)
      self.layer_norm2 = nn.LayerNorm(D)
      self.dropout = nn.Dropout(0.1)

    def forward(self, x):
      att = self.multivariate_attention(x,x,x)
      x = self.layer_norm1(x + self.dropout(att))
      #print("x_norm  : ", x.shape)
      #print("permute : ", xT.shape)
      x_forward = self.feed_forward(x)
      x= self.layer_norm2(x + x_forward)
      return x


class iTransformer(nn.Module):
    def __init__(self, N, T, D, S, proj_dim, num_blocks, use_norm=True):
      super(iTransformer, self).__init__()

      self.embedding = Embedding_inverted(T, D)
      self.trmblock = nn.ModuleList([TrmBlock(N, D, proj_dim) for _ in range(num_blocks)])
      self.projection = nn.Sequential(nn.Linear(D, D*4, bias=True), 
                                      nn.GELU(), 
                                      nn.Dropout(0.1),
                                      nn.Linear(D*4, S),
                                      nn.Dropout(0.1),
                                      )
      self.use_norm = use_norm
      self.S = S
      self.norm = nn.LayerNorm(D)
        
    def forward(self, x):
      #print("x : ",x.shape)
      if self.use_norm :
          means = x.mean(1, keepdim=True).detach()
          x = x - means
          std = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
          x/=std
            
      x = self.embedding(x)
      #print('emb : ',x.shape)
      for block in self.trmblock:
            x = block(x)
            
      x = self.norm(x)
      #print('trmblock : ',x.shape)
      y = self.projection(x)
      #print('proj : ', y.shape)
      y=y.permute(0,2,1)
      #print('final : ', y.shape)
      
      if self.use_norm : 
            y = y * (std[:,0,:].unsqueeze(1).repeat(1, self.S, 1))
            y = y + (means[:,0,:].unsqueeze(1).repeat(1, self.S, 1))
      return y[:, -self.S:, :]