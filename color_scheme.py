import random

col_set=[(0.184,0.413,1),(1,0.302,0.227),(0.562,0.725,0.248),(1,0.502,0.047),(0.878,0.195,0.553),(0.132,0.334,0.073),(0.406,0.366,1),(0.427,0.185,1)]
col_offset='BlackMyth:Wukong'

def get_color(seed, offset=''):
    seed = str(seed)
    random.seed(str(col_offset)+f'_{str(offset)}'+seed)
    col_ind = random.randrange(0,len(col_set))
    col=col_set[col_ind]
    return col