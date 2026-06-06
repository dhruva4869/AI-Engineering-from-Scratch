"""
Dense Transformer ka Problem
Normal transformer mein maan lo model ke 100 billion parameters hain.
Jab bhi ek token process hota hai:
    Token -> Sare 100B parameters use honge
    Chahe input simple ho ya difficult, har token ko poore model se guzarna padta hai.
    Isliye agar model ko aur smart banana hai:
    100B -> 200B -> 400B -> 800B
    toh har token ka computation (FLOPs) bhi utna hi badhta jayega.
    Yahi compute wall hai.

MoE (Mixture of Experts) ka Idea
Ek bada FFN rakhne ke bajay usko bahut saare experts mein divide kar do.
Example:
256 experts
Aur ek router decide karega ki current token ko kaunse experts ke paas bhejna hai.
Token = "database"

Router:
✓ Expert 7
✓ Expert 42
✓ Expert 101
✓ Expert 230
...


Real Example

DeepSeek-V3:

671B total parameters
37B active parameters per token

Matlab model ke paas 671B knowledge store hai, lekin ek token process karte waqt sirf ~37B parameters activate hote hain.
"""

"""
Router kya karta hai?

Har token ke liye router ek chhota neural network hota hai.
"database" -> [0.2, -1.5, 0.8, ...]
Router is embedding ko input leta hai aur har expert ke liye score nikalta hai.
Agar 4 experts hain:

Expert 1 -> 0.2
Expert 2 -> 1.8
Expert 3 -> 0.5
Expert 4 -> 1.3

Fir top-k choose karta hai.
Agar k=2:
Expert 2
Expert 4

Router mathematically
scores = token_embedding × W_router then softmax
"""

"""
Workload divide kaise hota hai within experts
Training ke dauran router khud seekhta hai.

Initially random routing hoti hai.

Example:

Python tokens -> Expert 17
SQL tokens -> Expert 89
Math tokens -> Expert 201
Chinese tokens -> Expert 55

Backpropagation se router aur experts dono train hote hain.

Total Loss =
Language Modeling Loss
+
Load Balancing Loss
"""


"""
How to handle loss?
Router bhi train hota hai
Agar router repeatedly wrong expert choose kare:

Math token → Poetry expert
toh loss badhega.
Backpropagation router ko sikha deta hai ki agli baar better routing kare.


Capacity overflow problem
Ek aur issue:
1000 tokens
sab Expert 7 par jana chahte hain
Expert 7 ke paas limited capacity hai.
"""