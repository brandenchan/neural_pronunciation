import random
import numpy as np
import tensorflow as tf

PAD_CODE = 0
START_TOKEN = "<START>"
END_TOKEN = "<END>"
END_CODE = 2

class JointIterator:
    def __init__(self, iter_x, iter_y, time_major=True, auto_reset=True):
        self.iter_x = iter_x
        self.iter_y = iter_y
        assert iter_x.len == iter_y.len
        self.len = iter_x.len
        self.time_major = time_major
        self.auto_reset = auto_reset

    def next(self, num):
        X_raw = self.iter_x.next(num)
        Y_raw = self.iter_y.next(num)
        if not X_raw and not Y_raw:
            return None
        X_lens = np.asarray([len(seq) for seq in X_raw])
        Y_lens = np.asarray([len(seq) for seq in Y_raw]) - 1
        Y = pad(Y_raw)
        X = pad(X_raw)
        Y_in = np.asarray(Y)[:, :-1]
        Y_targ = np.asarray(Y)[:, 1:]
        X = np.stack(X)


        # Y_in = np.stack(Y_in)
        # Y_targ = np.stack(Y_targ)

        if self.time_major:
            X = X.T
            Y_in = Y_in.T
            Y_targ = Y_targ.T

        return X, Y_in, Y_targ, X_lens, Y_lens

    def reset(self):
        self.iter_x.reset()
        self.iter_y.reset()

        
class DataIterator:
    """ This class modifies a regular iterator so that
    it will start iterating back at the beginning when it
    has reached the end """
    def __init__(self, data, auto_reset=True):
        self.auto_reset = auto_reset
        self.save = data
        self.len = len(data)
        self.iterator = iter(self.save)

    def next(self, num):
        ret = []
        for _ in range(num):
            try:
                ret.append(next(self.iterator))
            except StopIteration:
                if self.auto_reset:
                    self.reset()
                    ret.append(next(self.iterator))
                else:
                    return ret
        return ret

    def reset(self):
        self.iterator = iter(self.save)

def joint_iterator_from_file(filename, auto_reset=True, time_major=True):
    X, Y = load_cmu(filename)
    x_iter = DataIterator(X, auto_reset=auto_reset)
    y_iter = DataIterator(Y, auto_reset=auto_reset)
    return JointIterator(x_iter, y_iter, time_major=time_major)

def pad(seq_of_seq, pad_code=PAD_CODE):
    lens = np.asarray([len(s) for s in seq_of_seq])
    max_len = max(lens)
    pads_required = max_len - lens
    for i, s in enumerate(seq_of_seq):
        for _ in range(pads_required[i]):
            s.append(pad_code)
    return seq_of_seq

def load_cmu(filename, start_end=True, encoding=True):
    file = open(filename, encoding="utf-8")
    spell_to_code, _ = load_maps("data/chars")
    arpa_to_code, _ = load_maps("data/arpabet")
    spellings = []
    phonetics = []
    for l in file:
        if not l[0].isalpha():
            continue
        cols = l[:-1].split()
        if len(cols) < 2:
            continue
        spell = cols[0]
        phon = cols[1:]
        if start_end:
            spell = add_tokens(spell)
            phon = add_tokens(phon)      
        spellings.append(spell)
        phonetics.append(phon)
    assert len(spellings) == len(phonetics)
    return convert_list(spellings, spell_to_code), convert_list(phonetics, arpa_to_code)

def add_tokens(seq, start=START_TOKEN, end=END_TOKEN):
    ret = [start] + list(seq)
    ret.append(end)
    return ret

def convert_list(list, map):
    ret = []
    for word in list:
        ret.append(convert_word(word, map))
    return ret

def convert_word(word, map):
    ret = []
    for ch in word:
        ret.append(map[ch])
    return ret

def load_maps(sym_file):
    syms = load_symbols(sym_file)
    return create_map(syms)

def load_symbols(filename):
    return [sym[:-1] for sym in open(filename)]

def create_map(symbols):
    sym_to_code = {sym : i for i, sym in enumerate(symbols)}
    code_to_sym = {i : sym for i, sym in enumerate(symbols)}
    return sym_to_code, code_to_sym

