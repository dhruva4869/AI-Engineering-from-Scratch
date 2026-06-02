"""
Scaled dot-product attention is order-blind. The attention matrix softmax(Q K^T / √d) V is computed from pairwise similarities. 
Shuffle the rows of X, get the rows of the output shuffled the same way. Nothing inside attention cares about position.

The fix is to inject position into the embeddings somehow. Three eras of answers:
Absolute sinusoidal (Vaswani 2017). 
    Add sin/cos of position to the embedding. Simple, learnable-free, extrapolates poorly beyond trained lengths.

RoPE — Rotary Position Embeddings (Su 2021). 
    Rotate Q and K vectors by an angle proportional to position. Encodes relative position directly in the dot product. Dominant in 2026.

ALiBi — Attention with Linear Biases (Press 2022). 
    Skip embeddings entirely; add a per-head linear penalty to attention scores based on distance. Excellent length extrapolation.

As of 2026, essentially every frontier open model uses RoPE: Llama 2/3/4, Qwen 2/3, Mistral, Mixtral, DeepSeek-V3, Kimi. A handful of long-context models use ALiBi or its modern variants. Absolute sinusoidal is historical.

sin/cos added the angle value small part to the embedding, but that is very less, so we directly rotate the query value vectors itself
"""

"""
Absolute sinusoidal
PE[pos, 2i] = sin(pos / 10000 ^ (2i/d))
PE[pos, 2i+1] = cos(pos / 10000 ^ (2i/d))
"""
import math
def sinusodial(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2*i/d))
            pe[pos][2*i] = math.sin(theta)
            pe[pos][2*i+1] = math.cos(theta)
    return pe


"""
RoPE Rotary Position Embeddings 
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
"""
def rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d//2):
        theta = pos / (base ** (2*i/d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2*i], x[2*i+1]
        out[2*i] = a * c - b * s
        out[2*i+1] = a * s + b * c
    return out

"""
ALiBi - Skip the embedding trick. Bias the attention scores directly:
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
"""
def alibi(n_heads, seq_len):
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias


"""
Use It
"""
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192} # yarn + rope yet another rope extension
