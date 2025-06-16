import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import math

args = {'n': 4,
        'L': 4,
        'D': 512,
        'mlp_dim': 1024,
        'input_size': 1024,
        'dropout_rate': 0.}



class Mlp(nn.Module):
    def __init__(self, dim=args['D']):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(dim, dim*4)
        self.fc2 = nn.Linear(dim*4, dim)
        self.act_fn = torch.nn.functional.gelu  # torch.nn.functional.relu
        self.dropout = nn.Dropout(args['dropout_rate'])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, hidden_dim=args['D']):
        super(Block, self).__init__()
        self.attention_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.ffn = Mlp(dim = hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.attn = Attention(dim = hidden_dim)

    def forward(self, x1, x2):
        identity = x1
        x1 = self.attention_norm(x1)
        x2 = self.norm(x2)
        x1 = self.attn(x1, x2)
        x1 = x1 + identity

        identity = x1
        x1 = self.ffn_norm(x1)
        x1 = self.ffn(x1)
        x1 = x1 + identity

        return x1



class Attention(nn.Module):
    def __init__(self, dim=args['D']):
        super(Attention, self).__init__()
        self.num_attention_heads = args['n']
        self.attention_head_size = int(dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(dim, self.all_head_size)
        self.key = nn.Linear(dim, self.all_head_size)
        self.value = nn.Linear(dim, self.all_head_size)

        self.out = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(args['dropout_rate'])
        self.proj_dropout = nn.Dropout(args['dropout_rate'])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, x1, x2):
        mixed_query_layer = self.query(x1)
        mixed_key_layer = self.key(x2)
        mixed_value_layer = self.value(x2)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output

#Patch Embedding
class Embeddings(nn.Module):
    def __init__(self, in_channels=3, hidden_size=args['D'], img_size=args['input_size']):
        super(Embeddings, self).__init__()
        n_patches = img_size * img_size
        k_size = 1
        self.patch_embeddings = nn.Conv2d(in_channels, out_channels=hidden_size, kernel_size=k_size, stride=k_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(args['dropout_rate'])

    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden, n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)
        # print(x.shape, self.position_embeddings.shape)
        embeddings = x + self.position_embeddings
        
        embeddings = self.dropout(embeddings)
        return embeddings



    
class SingleTransformer(nn.Module):
    def __init__(self, in_channels1, in_channels2, feat_size1, feat_size2, hidden_size=128):
        super(SingleTransformer, self).__init__()
        self.embed1 = Embeddings(in_channels1, hidden_size, feat_size1)
        self.embed2 = Embeddings(in_channels2, hidden_size, feat_size2)
        self.encoder = Block(hidden_size)
        self.conv = nn.Conv2d(hidden_size, in_channels1, kernel_size=1)

    def forward(self, x_l, x_g):
        embed1 = self.embed1(x_l)
        embed2 = self.embed2(x_g)  # (B, n_patch, hidden//2)        
        encoded = self.encoder(embed1, embed2)
        B, n_patch, hidden = encoded.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        encoded = encoded.permute(0, 2, 1)
        encoded = encoded.contiguous().view(B, hidden, h, w)
        encoded = self.conv(encoded)
        return encoded


