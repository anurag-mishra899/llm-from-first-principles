class Tokenizer:
    """Character level tokenizer"""
    def __init__(self, text):
        self.text = text
        self.chars = sorted(list(set(text)))
        #print(f"Unique characters: {self.chars}")
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
    
    def encode(self, text):
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])