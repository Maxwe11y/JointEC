# fixed positiong encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout=0.1, max_len=200):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos):
        #x = x + self.pe[:x.size(0), :]
        x = x + self.pe[pos, :]
        return self.dropout(x)

# learned position encoding
class LearnedPositionEncoding(nn.Embedding):
    def __init__(self,d_model, dropout = 0.1,max_len = 200):
        super().__init__(max_len, d_model)
        self.dropout = nn.Dropout(p = dropout)
     
    def forward(self, x, pos):
        weight = self.weight.data.unsqueeze(1)
        #x = x + weight[:x.size(0),:]
        x = x + weight[pos,:]
        return self.dropout(x)

# it is a text classification model with only encoder
class TextTransformer(nn.Module):

    def __init__(self, position_enc, d_model = 512, d_out = 100, nhead = 5, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1):
        super(TextTransformer, self).__init__()

        # Preprocess
        self.pos_encoder_src = position_enc(d_model=d_model)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dim_feedforward,dropout)
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer,num_encoder_layers,encoder_norm)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead
        self.transform = nn.Linear(d_model, d_out)


    def forward(self, src, position, src_mask = None,memory_mask = None,src_key_padding_mask = None,
                memory_key_padding_mask = None):

        # shape check
        if src.size(2) != self.d_model:
            raise RuntimeError("the feature number of src must be equal to d_model")

        # position encoding
        src = self.pos_encoder_src(src, position)
        memory = self.encoder(src)

        #return F.softmax(memory, dim=2)
        output = self.transform(memory).squeeze(0)
        return output


    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                # xavier_uniform_(p)
                nn.init.xavier_uniform_(p)