"""
LSTM and RNNs have problem that they are always sequential and hence multiple epochs become a problem
RNN relies on simply the previous input. It has some types like
    LSTM have input gate forget gate output gate and candidate memory to store as the next candidate.
    GRU is basically simpler LSTM with only 2 gates, update and reset gate
Gradient descent is used to miminize the cost function and then maximize the final result
These algorithms in higher depth suffer from the vanishing gradient problem meaning that the next iterations have basically no real backpropagation


!RNN sequential compute vs Transformer parallel attention
Recurrence as a bottleneck.
    An RNN computes h_t = f(h_{t-1}, x_t). Each step depends on the previous. You cannot compute h_5 before h_4. 
The inductive bias shift.
    RNNs assume locality and recency. Transformers assume nothing so every pair is a candidate for attention
"""


def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent