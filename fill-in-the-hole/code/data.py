import numpy as np 
from tokenizer import tokenizer_

def random_input(text: str, num_mask=2) -> str:
    tokenizer = tokenizer_()
    token = tokenizer.tokenize(text)
    
    rand_position = np.random.randint(0, len(token), num_mask)
    for rand in list(rand_position):
        token[rand] = "[MASK]"
    token = "".join(token)
    print_text = token.replace("[MASK]", "●")
    print(f"({print_text}) predict ● from")
    print("~"*100)
    return token
