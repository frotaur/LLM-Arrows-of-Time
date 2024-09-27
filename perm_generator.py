### PERMUTATION LIST : 
import random

forward = lambda att : [i for i in range(att)] # Forward
backwards = lambda att : [(att-i-1) for i in range(att)] # Backwards

middle_out = lambda end : [end//2+(-1)**i*((i+1)//2) for i in range(end) ] # Middle-out
end_in =lambda att :  [((1+(-1)**i)//2)*i//2 + (1-(-1)**i)//2*(att-i//2-1) for i in range(att)] # Equiv to reversed middle-out

even_odd = lambda att : [i for i in range(att) if i%2==0]+[i for i in range(att) if i%2==1] # Even-odd
odd_even = lambda att : [i for i in range(att) if i%2==1]+[i for i in range(att) if i%2==0] # Odd-even

half_back_forward = lambda att : [(att//2-1)-i for i in range(att//2)]+[att//2+i for i in range(att//2)] # Half-back-forward
half_forward_back = lambda att : [i for i in range(att//2)]+[(att-1)-i for i in range(att//2)] # Half-forward-back

def quarter_middle_out_interleave(att) :
    assert att%4==0, 'att length must be a multiple of 4'
    middle_out1 = middle_out(att//2)
    middle_out2 = [middi + att//2 for middi in middle_out1]

    return [middle_out1[i//2] if i%2==0 else middle_out2[i//2] for i in range(att)]

def quarter_middle_out_concat(att) :
    assert att%4==0, 'att length must be a multiple of 4'
    middle_out1 = middle_out(att//2)
    middle_out2 = [middi + att//2 for middi in middle_out1]

    return middle_out1+middle_out2

def quarter_end_in_interleave(att) :
    assert att%4==0, 'att length must be a multiple of 4'
    end_in1 = end_in(att//2)
    end_in2 = [endini + att//2 for endini in end_in1]

    return [end_in1[i//2] if i%2==0 else end_in2[i//2] for i in range(att)]

def quarter_end_in_concat(att) :
    assert att%4==0, 'att length must be a multiple of 4'
    end_in1 = end_in(att//2)
    end_in2 = [endini + att//2 for endini in end_in1]

    return end_in1+end_in2

def uniform_random(att, seed):
    random.seed(seed)
    return random.sample(range(att), att)

def random_chunk(att, seed, chunk_size=8):
    assert att%chunk_size==0, 'Number of chunks must be a multiple of the sequence length'
    random.seed(seed)
    chunk_order = random.sample(range(att//chunk_size), att//chunk_size)
    return [chunk_order[ch_num]*chunk_size +i for ch_num in range(att//chunk_size) for i in range(chunk_size) ]