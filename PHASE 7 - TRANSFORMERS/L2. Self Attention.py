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




"""
Chalo ek simple sentence se samajhte hain:

**Sentence:**

```text
"The animal didn't cross the street because it was tired"
```

Model jab word **"it"** pe aata hai, usko samajhna hai ki **"it" kiski taraf refer kar raha hai?**

---

## Step 1: Har word ka embedding

Maan lo:

| Word    | Embedding |
| ------- | --------- |
| The     | x₁        |
| animal  | x₂        |
| didn't  | x₃        |
| cross   | x₄        |
| street  | x₅        |
| because | x₆        |
| it      | x₇        |
| was     | x₈        |
| tired   | x₉        |

Har embedding se 3 vectors bante hain:

```text
Q = Query
K = Key
V = Value
```

using learned matrices:

[
Q = XW_q
]

[
K = XW_k
]

[
V = XW_v
]

---

## Step 2: Query kya hai?

Jab hum **"it"** process kar rahe hain:

```text
Query(it)
```

basically pooch raha hai:

> "Mujhe batao main kis word se information loon?"

Think:

```text
Query = What am I looking for?
```

---

## Step 3: Key kya hai?

Har word apna key deta hai.

```text
animal -> Key_animal
street -> Key_street
tired -> Key_tired
...
```

Key ko socho:

```text
Key = Main kis type ki information provide karta hu?
```

---

## Step 4: Query aur Keys compare

Model compute karta hai:

[
score = Q_{it} \cdot K_j
]

for every word.

Example:

| Word    | Score |
| ------- | ----- |
| animal  | 8.5   |
| street  | 0.4   |
| because | 0.1   |
| tired   | 3.2   |

Highest score:

```text
animal = 8.5
```

Matlab:

> "it" ko lag raha hai "animal" important hai.

---

## Step 5: Softmax

Scores ko probabilities mein convert karte hain:

| Word   | Attention Weight |
| ------ | ---------------- |
| animal | 0.85             |
| tired  | 0.10             |
| street | 0.02             |
| others | 0.03             |

---

## Step 6: Value kya hai?

Ab actual information Value mein stored hai.

```text
animal -> V_animal
street -> V_street
...
```

Value ko socho:

```text
Value = Mere paas actual information hai
```

---

## Step 7: Weighted Sum

[
Output =
0.85V_{animal}
+
0.10V_{tired}
+
...
]

Since animal got most attention:

```text
Output ≈ V_animal
```

---

## Real-life analogy

Suppose:

```text
Query = "I need a database expert"
```

People in room:

| Person  | Key        | Value         |
| ------- | ---------- | ------------- |
| Alice   | database   | her knowledge |
| Bob     | frontend   | his knowledge |
| Charlie | networking | his knowledge |

Query compares against all Keys.

Best match:

```text
database ↔ database
```

Then you take Alice's Value (actual knowledge).

---

### One-line summary

For a token:

* **Query** = "Mujhe kya dhoondhna hai?"
* **Key** = "Main kis type ki information represent karta hu?"
* **Value** = "Mere paas actual information kya hai?"

Self-attention is basically:

```text
Query asks:
"Who should I listen to?"

Keys answer:
"Listen to me if you need this kind of information."

Values provide:
"The actual information."
```

That's why attention score uses **Query × Key**, but the information that gets passed forward comes from **Value**.

"""