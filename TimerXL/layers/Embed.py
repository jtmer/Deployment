import math
import torch
import torch.nn as nn
from torch.jit import is_scripting
from TimerXL.models.configuration_timer import TimerxlConfig
    
class TimerPatchEmbedding(nn.Module):
    def __init__(self, config: TimerxlConfig):
        super().__init__()
        self.input_token_len = config.input_token_len
        self.emb = nn.Linear(config.input_token_len,
                             config.hidden_size, bias=False)

    def forward(self, hidden_state: torch.Tensor):
        # hidden_state = hidden_state.unfold(
        #     dimension=-1, size=self.input_token_len, step=self.input_token_len)
        # return self.emb(hidden_state
        seq_length = hidden_state.shape[-1]
        num_patches = seq_length // self.input_token_len
        hidden_state = hidden_state.reshape(*hidden_state.shape[:-1], num_patches, self.input_token_len)
        return self.emb(hidden_state)
    
class TimeMoeRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=10000, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.max_seq_len_cached: int = 0
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim,
                          2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = int(seq_len)
        t = torch.arange(self.max_seq_len_cached, device=device,
                         dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        if not is_scripting():
            self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
            self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)
        else:
            self.cos_cached = emb.cos().to(dtype)
            self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, seq_len: int=0):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
