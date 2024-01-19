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
    def __init__(self, D, proj_dim, nb_head=8, get_attention=False):
        super(Attention, self).__init__()
        self.query_projection = nn.Linear(D, proj_dim)
        self.key_projection = nn.Linear(D, proj_dim)
        self.value_projection = nn.Linear(D, proj_dim)
        self.out_projection = nn.Linear(proj_dim, D)
        self.H = nb_head
        self.dropout = nn.Dropout(0.1)
        self.get_attention = get_attention

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
        A = torch.softmax(scale * scores, dim=-1)
        Att = self.dropout(A)
        V = torch.einsum("bhls,bshd->blhd", Att, values)

        out = V.reshape(B,L,-1)

        if self.get_attention : 
           A = A.transpose(2, 3).reshape(B, L, -1) 
           return self.out_projection(out), A
           
        return self.out_projection(out)
        

class TrmBlock(nn.Module):
    def __init__(self, N, D, proj_dim, get_attention=False):
      super(TrmBlock, self).__init__()

      self.multivariate_attention= Attention(D, proj_dim)
      self.layer_norm1 = nn.LayerNorm(D)
      self.feed_forward = FeedForward(D)
      self.layer_norm2 = nn.LayerNorm(D)
      self.dropout = nn.Dropout(0.1)

      self.get_attention = get_attention

    def forward(self, x):
      if self.get_attention :
          att, A = self.multivariate_attention(x,x,x)
      else : 
          att = self.multivariate_attention(x,x,x)

      x = self.layer_norm1(x + self.dropout(att))
      x_forward = self.feed_forward(x)
      x= self.layer_norm2(x + x_forward)
      
      if self.get_attention :
         return x, A
      return x


class TrmBlock_Att_Att(nn.Module): 
    def __init__(self, N, D, proj_dim):
      super(TrmBlock_Att_Att, self).__init__()

      self.attention_variate= Attention(D, proj_dim) #variate with attention
      self.layer_norm1 = nn.LayerNorm(D)
      self.attention_temporal = Attention(N, proj_dim) #temporal with attenton
      self.layer_norm2 = nn.LayerNorm(D)
      self.dropout = nn.Dropout(0.1)

    def forward(self, x):
      att = self.attention_variate(x,x,x)
      x = self.layer_norm1(x + self.dropout(att))

      # veux faire l'attention sur le temporal : sur N 
      x = x.permute(0,2,1)
      x_forward = self.attention_temporal(x, x, x)
      x = x.permute(0,2,1)
      x_forward = x_forward.permute(0,2,1)

      x= self.layer_norm2(x + x_forward)
      return x

class TrmBlock_FFN_Att(nn.Module): 
    def __init__(self, N, D, proj_dim):
      super(TrmBlock_FFN_Att, self).__init__()

      self.feedforward_variate= FeedForward(N) #variate with FFN
      self.layer_norm1 = nn.LayerNorm(D)
      self.attention_temporal = Attention(N, proj_dim) #temporal with attenton
      self.layer_norm2 = nn.LayerNorm(D)
      self.dropout = nn.Dropout(0.1)

    def forward(self, x):
      #veut faire le feedforward sur les variates : sur D
      x = x.permute(0,2,1)
      ffn = self.feedforward_variate(x)
      ffn = ffn.permute(0,2,1)
      x = x.permute(0,2,1)


      x = self.layer_norm1(x + self.dropout(ffn))

      # veux faire l'attention sur le temporal : sur N 
      x = x.permute(0,2,1)
      x_forward = self.attention_temporal(x,x,x)
      x = x.permute(0,2,1)
      x_forward = x_forward.permute(0,2,1)

      x= self.layer_norm2(x + x_forward)
      return x

class TrmBlock_FFN_FFN(nn.Module): 
    def __init__(self, N, D, proj_dim):
      super(TrmBlock_FFN_FFN, self).__init__()

      self.feedforward_variate= FeedForward(N) #variate with FFN
      self.layer_norm1 = nn.LayerNorm(D)
      self.feedforward_temporal = FeedForward(D) #temporal with FFN
      self.layer_norm2 = nn.LayerNorm(D)
      self.dropout = nn.Dropout(0.1)

    def forward(self, x):
      #veut faire l'attention sur les variates : sur D
      x = x.permute(0,2,1)
      ffn = self.feedforward_variate(x)
      ffn = ffn.permute(0,2,1)
      x = x.permute(0,2,1)


      x = self.layer_norm1(x + self.dropout(ffn))

      # veux faire le feedforward sur le temporal : sur N 
      x_forward = self.feedforward_temporal(x)

      x= self.layer_norm2(x + x_forward)
      return x  


class TrmBlock_Att_variate(nn.Module): 
    def __init__(self, N, D, proj_dim):
      super(TrmBlock_Att_variate, self).__init__()

      self.multivariate_attention= Attention(D, proj_dim)
      self.layer_norm1 = nn.LayerNorm(D)
      self.dropout = nn.Dropout(0.1)

    def forward(self, x):
      att = self.multivariate_attention(x,x,x)
      x = self.layer_norm1(x + self.dropout(att))
      return x


class TrmBlock_FFN_temporal(nn.Module): 
    def __init__(self, N, D, proj_dim):
      super(TrmBlock_FFN_temporal, self).__init__()

      self.feed_forward = FeedForward(D)
      self.layer_norm2 = nn.LayerNorm(D)
      self.dropout = nn.Dropout(0.1)

    def forward(self, x):
      x_forward = self.feed_forward(x)
      x= self.layer_norm2(x + x_forward)
      return x


class iTransformer(nn.Module):
    def __init__(self, N, T, D, S, proj_dim, num_blocks, use_norm=True, typeTrmBlock="inverted", get_attention=False):
      super(iTransformer, self).__init__()

      self.embedding = Embedding_inverted(T, D)

      self.trmblock = None
      if typeTrmBlock=="inverted" : 
          self.trmblock = nn.ModuleList([TrmBlock(N, D, proj_dim) for _ in range(num_blocks)])
      elif typeTrmBlock=="Att_Att" : 
          self.trmblock = nn.ModuleList([TrmBlock_Att_Att(N, D, proj_dim) for _ in range(num_blocks)])
      elif typeTrmBlock=="FFN_Att" : 
          self.trmblock = nn.ModuleList([TrmBlock_FFN_Att(N, D, proj_dim) for _ in range(num_blocks)])
      elif typeTrmBlock=="FFN_FFN": 
          self.trmblock = nn.ModuleList([TrmBlock_FFN_FFN(N, D, proj_dim) for _ in range(num_blocks)])
      elif typeTrmBlock=="Att_variate": 
          self.trmblock = nn.ModuleList([TrmBlock_Att_variate(N, D, proj_dim) for _ in range(num_blocks)])
      elif typeTrmBlock=="FFN_temporal": 
          self.trmblock = nn.ModuleList([TrmBlock_FFN_temporal(N, D, proj_dim) for _ in range(num_blocks)])


         
      self.projection = nn.Sequential(nn.Linear(D, D*4, bias=True), 
                                      nn.GELU(), 
                                      nn.Dropout(0.1),
                                      nn.Linear(D*4, S),
                                      nn.Dropout(0.1),
                                      )
      self.use_norm = use_norm
      self.S = S
      self.norm = nn.LayerNorm(D)

      self.liste_attention = []
      self.get_attention = get_attention
        
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
            if self.get_attention : 
                x, A = block(x)
                self.liste_attention.append(A)
            else : 
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
    

