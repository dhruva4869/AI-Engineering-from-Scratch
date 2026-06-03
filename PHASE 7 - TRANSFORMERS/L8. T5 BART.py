"""
**T5** ek encoder-decoder Transformer hai jahan encoder poora input text 
read karke uska contextual representation banata hai. 
Decoder output ko token-by-token generate karta hai aur har step par do 
jagah attention lagata hai: pehla apne pehle generate kiye gaye tokens par 
(masked self-attention) aur doosra encoder ke output par (cross-attention). 
T5 ki khas baat ye hai ki har task ko text-to-text format mein convert kiya jata hai, 
jaise translation, summarization, question answering sab text input → text output ban jata hai.

**BART** bhi encoder-decoder architecture use karta hai, lekin uska pretraining objective alag hai. 
Training ke dauran input text ko corrupt kiya jata hai (words mask karna, delete karna, sentence shuffle karna, etc.) 
aur encoder us corrupted text ko process karta hai. Decoder encoder ki representations ka use karke original text reconstruct 
karna seekhta hai. Isliye BART ko ek denoising autoencoder ki tarah samajh sakte ho, jabki T5 ko general-purpose text-to-text framework ki tarah.

"""
from transformers import T5Tokenizer, T5ForConditionalGeneration
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
text = "translate English to German: I love cricket"
inputs = tokenizer(text, return_tensors="pt")
output = model.generate(**inputs)
print(tokenizer.decode(output[0], skip_special_tokens=True))