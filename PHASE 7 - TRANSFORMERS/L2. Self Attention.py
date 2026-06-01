"""
Self-attention lets every position in a sequence attend to every other position in a single parallel step. That is what makes transformers fast, scalable, and dominant.
Why does vanilla self-attention scale the dot product by 1/sqrt(d_k)?
    To prevent dot products from growing large in high dimensions, which would push softmax into regions with tiny gradients
What are the three projections in self-attention?
    Query, key, and value -- each a learned linear projection of the same input
    queries (what am I looking for?), keys (what do I contain?), and values (what do I output if matched?).

Query, key
matmul
scaling
masking
softmax
and then matmul with value
do this multiple times as a scaled dot product attention and then do concat for multi head attention

Why scaling ->
Jab dk (dimensions) bada hota hai, to Q·K dot product (matmul) ki value bahut badi ho jaati hai.
Softmax phir almost 0 aur 1 output dene lagta hai, jisse gradients bahut chhote ho jaate hain.
Isliye score ko root(dk) se divide karte hain taaki values normal range mein rahein aur training stable rahe.

Why masking ->
Jisse token ko kuch aisa information na dikh jaye jisse uska outcome deviate / infer hone lage
Cheating nahi honi chahiye issliye masking required
issliye top triangle of matrix of mask krdiya jaate hai with -inf jisse state 1 cant look at 2, 2 cant look at 3 and so on
Diagonal mai i == j so that is ok since they can look at themselves for prediction.
matrix ij mtlb token i kitna attention dena chahte hai token j ko

Why softmax ->
Probablity function hai ye toh ekdm
Softmax converts raw scores into a probability distribution across each row:
"""

import numpy as np

def softmax(scores):
    """
    scores = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    Row 0: Token 0 looks at [Token0, Token1, Token2]
    Row 1: Token 1 looks at [Token0, Token1, Token2]
    """
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    """
    scores = [
    [-2, -1,  0],
    [-2, -1,  0]
    ]
    """
    exp_scores = np.exp(scores)
    """
    Exponent and then divide them each by row sum
    [
        [0.135, 0.368, 1.0],
        [0.135, 0.368, 1.0]
    ] DIVIDE BY
    [
        [1.503],
        [1.503]
    ]
    to get final softmax
    """
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


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
