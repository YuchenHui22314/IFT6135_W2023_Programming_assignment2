from encoder_decoder_solution import EncoderDecoder
import torch

ed = EncoderDecoder()
print(ed(torch.tensor([[1,1],[1,1]], dtype = torch.long), torch.tensor([[1,0],[1,0]], dtype=torch.long))[0].shape)

def shutong(a,b):
    return a+b

ddd = {"zyt":5,"yiyang":6}

print(shutong(**ddd))