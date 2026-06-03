"""
Language Model ka basic kaam bahut simple hai: agar pehle ke tokens diye hain, toh agla token kya hoga?
Problem ye hai ki training ke time poora sentence ek saath diya jata hai. 
Agar model future words dekh sake, toh cheating ho jayegi kyunki answer already saamne hai. 
Isliye causal mask use hota hai. Ye ek matrix hai jo har token ko sirf apne left side (past) ke tokens dekhne deta hai. 
Right side (future) ke tokens ko mask karke unki attention probability 0 kar di jaati hai. (ya -inf better)
Is trick ki wajah se model poori sequence ko parallel process kar sakta hai, lekin fir bhi future nahi dekh sakta. 
Isi architecture ko decoder-only transformer bolte hain.
Ye wohi hai jo decoder mai masked multi head attention hai and GPT use krta hai without encoders
"""

"""
      t1  t2  t3  t4
t1    ✓   0   0   0
t2    ✓   ✓   0   0
t3    ✓   ✓   ✓   0
t4    ✓   ✓   ✓   ✓
"""

"""
Model har position pe next token ki probability distribution nikalta hai. Hume usko batana hota hai ki prediction kitni sahi thi. Uske liye loss lagate hain.
Input:
I love eating

Actual next token:
pizza

Model predict karta hai:

pizza  = 0.7
burger = 0.2
pasta  = 0.1

Toh model ne sahi answer (pizza) ko high probability di hai, isliye loss kam hoga.

Agar predict kare:
pizza  = 0.05
burger = 0.9
pasta  = 0.05
Toh actual token pizza tha, lekin usko sirf 5% probability mili. Loss bahut bada hoga.
"""

"""
Loss=-log(P(correct token))
USe temlerature to spread out probabilities
or top k 
or semi-cheating for force learning some tokens
"""

