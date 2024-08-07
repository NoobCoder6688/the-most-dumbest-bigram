import torch

words = open('names.txt', 'r').read().splitlines()

N = torch.zeros((27,27),dtype=torch.int32)
alphabet = sorted(set(''.join(words)))
#create a dic
stoi = {s: i+1 for i, s in enumerate(alphabet)}
stoi['.'] = 0
print()
itos = {i: s for s, i in stoi.items()}

for word in words:
    chs =  ['.'] + list(word) + ['.']
    for char1, char2 in zip(chs, chs[1:]):
        ix1 = stoi[char1]
        ix2 = stoi[char2]
        N[ix1, ix2] += 1

g = torch.Generator().manual_seed(2147483647)

P = N.float()
P /= P.sum(1,keepdim=True)

Out = ''
for i in range(20):
    out = []
    ix = 0
    while True:
        p = P[ix]
        ix = torch.multinomial(p,num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix ==0:
            break
    print(''.join(out))



