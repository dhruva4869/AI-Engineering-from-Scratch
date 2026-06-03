"""
BERT (Google) - 
    Bunch of encoders only. No decoders
    Generates meaningful embeddings for words and entire setences and tokens
    eg - 
        Mark is a good person
        I only got 1 Mark
        Mark that problem as done please

        All 3 marks are different contextually and that can be identified by BERT
        It is trained on literally the entire internet
"""

"""
How was BERT trained?
Take a sentence: 
the quick brown fox jumps over the lazy dog.

Mask 15% of tokens randomly:
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog

Train the model to predict the original tokens at masked positions. 
Because the encoder is bidirectional, predicting [MASK] at position 1 can use brown fox jumps at positions 2+. That is the thing GPT cannot do.
"""

"""
BERT Masking rules
Of the 15% of tokens selected for prediction:

80% are replaced with [MASK].
10% are replaced with a random token.
10% are left unchanged.

Why not 100% mask of the 15% this is for fine tuning
"""

"""
BERT vs MODERNBERT (2024/2026)

Positional Encoding  ---------      sin/cos vs ROPE
Activation           ---------      Gelu vs GeGlu
Normalization        ---------      LayerNorm vs RMSNorm
Tokenizer            ---------      WordPiece vs BPE
"""


from transformers import AutoModel, AutoTokenizer
tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")
text = "Hello world"
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state