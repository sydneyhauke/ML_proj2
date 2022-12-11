import numpy as np
import string
from permutation import Permutation

class BinaryOperator:
    def __init__(self, op, symbol:str):
        self.op     = np.vectorize(op)
        self.symbol = symbol
        
    def apply(self, x, y):
        return self.op(x, y)

p = 97
S5 = list(Permutation.group(5))

## UTILS 
np_pow_mod = np.frompyfunc(pow, 3, 1)
def get_lehmer(n):
    return np.vectorize(lambda perm: perm.lehmer(n))
## #####

hebrew_codes = np.arange(0x05D0, 0x05EA)
greek_codes  = np.arange(0x03b1, 0x03c9) #lowercase only
extended_latin_codes = np.arange(0x0100, 0x0148)

fancy_chr = np.vectorize(chr)

ALL_SYMBOLS   = list(string.ascii_lowercase + string.ascii_uppercase) + list(fancy_chr(greek_codes)) + list(fancy_chr(hebrew_codes)) + list(fancy_chr(extended_latin_codes))
TOTAL_SYMBOLS = len(ALL_SYMBOLS)

## Operations in Zp (p=97)
addition_mod_p           = BinaryOperator(lambda x, y: (x + y) % p, "+")
subtraction_mod_p        = BinaryOperator(lambda x, y: (x - y) % p, "-")
division_mod_p           = BinaryOperator(lambda x, y: (x * np_pow_mod(y, -1, p)) % p, "/")
div_or_sub_mod_p         = BinaryOperator(lambda x, y: (x * np_pow_mod(y, -1, p)) %p if (y % 2 == 1) else (x-y) %p, "(*1)")
sum_of_squares_mod_p     = BinaryOperator(lambda x, y: (np_pow_mod(x, 2, p) + np_pow_mod(y, 2, p)) %p, "(*2)")
sum_of_mixed_terms_mod_p = BinaryOperator(lambda x, y: (np_pow_mod(x, 2, p) + x*y + np_pow_mod(y, 2, p)) %p, "(*3)")
fancy_sum_mod_p_1        = BinaryOperator(lambda x, y: (np_pow_mod(x, 2, p) + x*y + np_pow_mod(y, 2, p) + x) %p, "(*4)")
fancy_sum_mod_p_2        = BinaryOperator(lambda x, y: (np_pow_mod(x, 3, p) + x*y) %p, "(*5)")
fancy_sum_mod_p_3        = BinaryOperator(lambda x, y: (np_pow_mod(x, 3, p) + x*np_pow_mod(y, 2, p) + y) %p, "(*6)")

## Operations in Sn (n=5)
composition              = BinaryOperator(lambda x, y: x * y, "*7")
composition_with_inverse = BinaryOperator(lambda x, y: x * y * x.inverse(), "*8")
composition_with_x       = BinaryOperator(lambda x, y: x * y * x, "*9")

def make_dict(size):
    ints   = np.arange(size)
    labels = ALL_SYMBOLS[:size]
    
    return dict(zip(ints, labels))

def make_dict_v2(size):
    np.random.seed(88)
    ints   = np.arange(size)
    labels = np.arange(size)
    
    np.random.shuffle(labels)
    
    return dict(zip(ints, labels))

def vectorized_token_transform(i2token: dict):
    return np.vectorize(i2token.get)

def transform_numeric_results(op, eq, i2str: dict):
    return np.vectorize(lambda x, y, res: f"{i2str[x]}{op}{i2str[y]}{eq}{i2str[res]}")

def create_dataset(op: BinaryOperator, eq, x, y, i2str: dict, p, inS5=False):    
    xv, yv = np.meshgrid(x, y)
    
    numeric_results = op.apply(xv, yv)
    
    if (inS5): ## translate permutations to their Lehmer code; can now assign labels for p elts
        lehmer_transform = get_lehmer(p)
        numeric_results = lehmer_transform(numeric_results)
        
        xv = lehmer_transform(xv)
        yv = lehmer_transform(yv)
    
    fn = transform_numeric_results(op.symbol, eq, i2str)
    
    return fn(xv, yv, numeric_results)

def create_dataset_v2(op: BinaryOperator, x, y, i2token: dict, p, inS5=False):
    xv, yv = np.meshgrid(x, y)
    
    numeric_results = op.apply(xv, yv)
    
    if (inS5): ## translate permutations to their Lehmer code; can now assign labels for p elts
        lehmer_transform = get_lehmer(p)
        numeric_results = lehmer_transform(numeric_results)
        
        xv = lehmer_transform(xv)
        yv = lehmer_transform(yv)
    else:
        dict_transform = vectorized_token_transform(i2token)
        
        numeric_results = dict_transform(numeric_results)
        
        xv = dict_transform(xv)
        yv = dict_transform(yv)

        vocab_size = len(i2token)
        
    ds = np.dstack((xv, yv))
    ds_size = ds.size
    
    return np.dstack((xv, yv)).reshape((ds_size//2, 2)), numeric_results.reshape((ds_size//2)), vocab_size
