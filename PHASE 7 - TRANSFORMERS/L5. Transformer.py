"""
The 2017 Vaswani paper packaged six design decisions that turned one attention layer into a stackable block. 
Every transformer since — encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5) — inherits the same skeleton. 
In 2026 the blocks have been refined (RMSNorm, SwiGLU, pre-norm, RoPE) but the skeleton is identical.

Feature vector mtlb running embedding jo chal raha apna
Q, K, V token ke saare features (hidden state / feature vector) se bante hain.

Maan lo token "cricket" ka hidden state hai:
h = [0.2, 1.5, -0.7, 0.9]
Ye 4 features hain.
Ab:
Q = h x Wq
K = h x Wk
V = h x Wv

LayerNorm --> feature vectors ke layer ka mean
RMSNorm --> feature vectors ke layer ka RMS (ROOT MEAN SQUARE)
"""

"""
Encoder: Input ko poora padh ke uska meaning/context samajhta hai.
Decoder: Us meaning ko use karke output ek-ek token generate karta hai like for translation

Input:
"I love cricket"

Encoder:
"Accha, ye sentence cricket pasand hone ke baare mein hai."
Decoder:
"Mujhe cricket pasand hai"

Bas simple terms mein:
Encoder = Samajhne wala
Decoder = Likhne/Generate karne wala
"""

"""
Encoder Architecture

Input Embeddings --> Positional Encodings (Yarn) --> MHA --> Add & Norm (LayerNorm) --> FFN (Feed Forward Network) --> Add & Norm --> Decoder
"""

"""
Decoder Architecture

Output Embeddings --> PE --> Masked MHA --> Add & Norm --> MHA (output from Encoder) --> Add & Norm --> FFN --> Add & Norm
"""

"""
Output from Encoder seedhe MHA mai kyu aata hai and Masked MHA mai kyu nahi?
Dekho Output Embedding start mai toh null hoga but baad mai toh running hoga. Like "Mujhe cricket...."
Ab decoder ko nahi pata ki next word is "pasand hai" but usko apna context pata hai.
Abhi Encoder ne pasand related values ko encode krke bhej diya hai and top of the masked MHA of existing running output the decoder already has
Isse ab woh apna new output generate krne lagega
"""

"""
Input -> PE -> MHA -> AN -> FFN -> AN
Output -> PE -> MMHA -> AN ->MHA (from up) ->FFN->AN ->next
"""




"""
Encoder
I like football  ---> numerical  (Decoder)  -->  Mujhe football pasand hai

Encoder
Input embeddings   ---> PE  -----> MHA ------> Add & Norm  -------> FFN ----> Add & Norm

Decoder
Output embeddings   ---> PE  -----> Masked MHA ------> Add & Norm  -------> MHA ----> Add & Norm ----> FFN ----> Add & Norm

"""