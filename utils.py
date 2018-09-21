import numpy as np

def convert_list(list, map):
    ret = []
    for word in list:
        ret.append(convert_word(word, map))
    return ret

def convert_word(word, map):
    ret = []
    for ch in word:
        ret.append(map[ch])
    return np.asarray(ret)

def load_maps(sym_file):
    syms = load_symbols(sym_file)
    return create_map(syms)

def load_symbols(filename):
    return [sym[:-1] for sym in open(filename)]

def create_map(symbols):
    sym_to_code = {sym : i for i, sym in enumerate(symbols)}
    code_to_sym = {i : sym for i, sym in enumerate(symbols)}
    return sym_to_code, code_to_sym

