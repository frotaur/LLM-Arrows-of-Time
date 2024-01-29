from modules import TokenTexth5
from modules import tokenizer
import torch


toke = tokenizer.get_tokenizer(m_name='gpt2')
toke_mine = tokenizer.get_tokenizer(m_path='hellas/el_tokenizer',m_name='greek')
data = TokenTexth5('hellas/hellas.h5',40,backwards=True)
print(data[1][0])
phrase = 'Γαμώ την πουτάνα σου'
answer = toke.detokenize(data[1][1][None,:].to(torch.long))

for toki in [toke,toke_mine]:
    print('Toke length : ', toki.tokenize(phrase).shape[1])
print(f'data : {toke.detokenize(toke.tokenize(phrase))}, answer : {toke.detokenize(toke.tokenize(answer))}')
print(f'data : {toke_mine.detokenize(toke_mine.tokenize(phrase))}, answer : {toke_mine.detokenize(toke_mine.tokenize(answer))}')