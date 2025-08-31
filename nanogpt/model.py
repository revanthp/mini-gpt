"""nano-GPT implementation"""
import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from loguru import logger

@dataclass
class GPTConfig:
    block_size:int
    vocab_size:int
    n_head:int
    n_layer:int
    n_embeddings:int
    dropout:float
    bias:bool
    n_layers:int

class LayerNorm(nn.Module):
    def __init__(self, ndim:int, bias:bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



class CausalSelfAttention(nn.Module):
    def __init__(self, config:GPTConfig):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embeddings, 3 * config.n_embeddings, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embeddings, config.n_embeddings, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embeddings = config.n_embeddings
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        logger.info(f"flash is enabled: {self.flash}")

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embeddings)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embeddings, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embeddings, 4 * config.n_embeddings, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embeddings, config.n_embeddings, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embeddings, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embeddings, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config:GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict({
            'wte': nn.Embedding(self.config.vocab_size, self.config.n_embeddings),
            'wpe': nn.Embedding(self.config.block_size, self.config.n_embeddings),
            'dropout': nn.Dropout(self.config.dropout),
            'hidden': nn.ModuleList([Block(self.config) for _ in range(self.config.n_layers)]),
            'layer_norm': LayerNorm(self.config.n_embeddings, bias=self.config.bias)
            }
        )
        self.lm_head = nn.Linear(self.config.n_embeddings, self.config.vocab_size, bias=False)
        self.transformer['wte'].weight = self.lm_head.weight
        self.apply(self._init_weights)
        logger.info(f"No. of Parameters: {self.get_num_params()}")

    def _init_weights(self, module, mean:float=0.0, std:float=0.2) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=mean, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=mean, std=std)

    def get_num_params(self, skip_embeddings:bool=True) -> int:
        n_params = sum(layer.numel() for layer in self.parameters())
        if skip_embeddings:
            n_params -= self.transformer['wpe'].weight.numel()
        return n_params

    def forward(self, token_id, targets=None):
        token_emb = self.transformer['wte'](token_id)
        pos_emb = self.transformer['wpe'](token_id)
        x = self.transformer['dropout'](token_emb + pos_emb)
        for block in self.transformer['hidden']:
            x = block(x)
        x = self.transformer['layer_norm'](x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, -1, :])
            loss = None

        return logits, loss

    @torch.no_grad
    def generate(self, token_ids, max_new_tokens:int, temperature:float, top_k:int):
        for _ in range(max_new_tokens):
            cond = token_ids if token_ids.size(1) < self.config.block_size else token_ids[:, -self.config.block_size:]
            logits, _ = self(token_ids)
            logits = logits[:,-1,:] / temperature
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_ids = torch.cat((token_ids, next_token), dim = 1)
        return token_ids

    def configure_optimizer(self, learning_rate:float, weight_decay:float, betas):

        params = {pn:p for pn, p in self.named_parameters() if p.requires_grad}
        optim_groups = [
            {  # decay params
                'params':[p for _, p in params.items() if p.dim() >= 2],
                'weight_decay': weight_decay,
            },
            {  # no decay params
                'params':[p for _, p in params.items() if p.dim() < 2],
                'weight_decay': 0.0
            }
        ]

        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
