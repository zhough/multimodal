from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3ForCausalLM,
)
import torch 
from torch import nn
from torch.nn import functional as F

class CrossAttention(nn.Module):
    def __init__(self,v_hidden_size,l_hidden_size,num_heads):
        super().__init__()

        self.v_hidden_size = v_hidden_size
        self.l_hidden_size = l_hidden_size
        self.hidden_size = self.l_hidden_size   #中间层的维度和llm的隐藏层维度一致
        self.num_heads = num_heads
        self.head_dim = self.hidden_size//self.num_heads

        #线性映射层
        self.q_proj = nn.Linear(self.l_hidden_size,self.hidden_size)
        self.k_proj = nn.Linear(self.l_hidden_size,self.hidden_size)
        self.v_proj = nn.Linear(self.v_hidden_size,self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size,self.l_hidden_size)

    def forward(self,v_hidden_states,l_hidden_states):
        batch_size,l_seq_len,_ = l_hidden_states.shape  
        _,v_seq_len,_ = v_hidden_states.shape   
        q = self.q_proj(l_hidden_states)
        k = self.k_proj(v_hidden_states)
        v = self.v_proj(v_hidden_states)
        #转为多头
        q = q.reshape(batch_size,l_seq_len,self.num_heads,self.head_dim).transpose(1,2)
        k = k.reshape(batch_size,v_seq_len,self.num_heads,self.head_dim).transpose(1,2)
        v = v.reshape(batch_size,v_seq_len,self.num_heads,self.head_dim).transpose(1,2)

        # q = q.reshape(-1,l_seq_len,self.head_dim)
        # k = k.reshape(-1,v_seq_len,self.head_dim)
        # v = v.reshape(-1,v_seq_len,self.head_dim)

        attn_weight = torch.matmul(q,k.transpose(-1,-2))/torch.sqrt(self.hidden_size)
        attn_weight = F.softmax(attn_weight,dim=-1)

        attn_output = torch.matmul(attn_weight,v)
        attn_output = attn_output.transpose(1,2).reshape(batch_size,l_seq_len,self.hidden_size)
        attn_output = self.out_proj(attn_output)
        return attn_output  







