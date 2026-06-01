"""
!Multi-head attention splits, attends, concatenates
Split. Take X of shape (N, d_model). 
Project to Q, K, V each of shape (N, d_model). Reshape to (N, n_heads, d_head) where d_head = d_model / n_heads. Transpose to (n_heads, N, d_head).
Attend in parallel
Concatenate and project
"""
import numpy as np
def softmax(scores):
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def scaled_dot_product_attention(Q, K, V):
    """
    Matmul, scaling, masking, softmax process
    """
    dk = Q.shape[-1]
    scaling = Q @ K.T / np.sqrt(dk)
    weights = softmax(scaling)
    output = weights @ V
    return output, weights



class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk
    
    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights


# ------------------------------------ MHA ------------------------------------


def split_heads(X, n_heads):
    n, d = X.shape
    """
    X =
    [
        [1, 2, 3, 4],      # token 1
        [5, 6, 7, 8],      # token 2
        [9,10,11,12]       # token 3
    ]
    converted to this
    [
        [
            [1,2],     # head0 part of token1
            [3,4]      # head1 part of token1
        ],

        [
            [5,6],
            [7,8]
        ],

        [
            [9,10],
            [11,12]
        ]
    ]
    but now we want all head ka data eksaath so do transpose 1, 0, 2 so head, n, d_head

    [
        # head0
        [
            [1,2],
            [5,6],
            [9,10]
        ],

        # head1
        [
            [3,4],
            [7,8],
            [11,12]
        ]
    ]
    """
    # number of tokens, embedding dimension
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2) # (heads, n, d_head)



def combine_heads(H):
    # Splits ka reverse operation
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)


def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights




# ------------------------------------ TODO pratice ------------------------------------


def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)

def combine_heads(H):
    h, n, d_heads = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_heads)

def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.tranpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    output = weights @ Vh
    output = combine_heads(output)
    return output @ W_o, weights