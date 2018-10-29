""" Contains functions, classes and constants used to load CMU data from file
and create custom iterators to be used by the CharToPhonModel class. """

import random
import numpy as np
import tensorflow as tf

PAD_CODE = 0
START_TOKEN = "<START>"
END_TOKEN = "<END>"
END_CODE = 2

class JointIterator:
    """ This class iterates through two iterators simultaneously returning batches of data. """

    def __init__(self, iter_x, iter_y, time_major=True, auto_reset=True, extend_incomplete=False):
        """ 
        Args:
            auto_reset (bool): If True, the JointInterator will start again at the
                                beginning of the data set once it reaches the end. If
                                False, JointIterator.next() will return None
            extend_incomplete (bool): If True, dummy examples will be added to the batch
                                      when there are not enough examples left in the iterator
                                      to create a full batch. If False JointIterator.next()
                                      will return None which will signal that it is at the
                                      end of the data set.
        """

        self.iter_x = iter_x
        self.iter_y = iter_y
        assert iter_x.len == iter_y.len
        self.len = iter_x.len
        self.time_major = time_major
        self.auto_reset = auto_reset    
        self.extend_incomplete = extend_incomplete

    def next(self, num):
        """ Return num many data examples in a dictionary """
        X_raw = self.iter_x.next(num)
        Y_raw = self.iter_y.next(num)
        if not X_raw and not Y_raw:
            return None
        if not self.extend_incomplete:
            if len(X_raw) != num:
                return None

        len_X = np.asarray([len(seq) for seq in X_raw])
        len_Y = np.asarray([len(seq) for seq in Y_raw]) - 1
        Y = pad(Y_raw)
        X = pad(X_raw)
        Y_in = np.asarray(Y)[:, :-1]
        Y_targ = np.asarray(Y)[:, 1:]
        X = np.stack(X)

        diff = num - X.shape[0]

        if self.time_major:
            X = X.T
            Y_in = Y_in.T
            Y_targ = Y_targ.T

        batch = {"X": X,
                "Y_in": Y_in,
                "Y_targ": Y_targ,
                "len_X": len_X,
                "len_Y": len_Y}

        # If a batch is fetched with less examples than num
        if diff:
            batch = extend_batch(batch, num)

        return batch, diff

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

def joint_iterator_from_file(filename, auto_reset=True, time_major=True, n=None):
    """ Return a JointIterator by specifying the name of a CMU data file """
    X, Y = load_cmu(filename)
    if n:
        X = X[:n]
        Y = Y[:n]
    x_iter = DataIterator(X, auto_reset=auto_reset)
    y_iter = DataIterator(Y, auto_reset=auto_reset)
    return JointIterator(x_iter, y_iter, time_major=time_major)

def pad(seq_of_seq, pad_code=PAD_CODE):
    """ Pad sequences in seq_of_seq with pad_code so that
    each sequence is of the same length """

    lens = np.asarray([len(s) for s in seq_of_seq])
    max_len = max(lens)
    pads_required = max_len - lens
    for i, s in enumerate(seq_of_seq):
        for _ in range(pads_required[i]):
            s.append(pad_code)
    return seq_of_seq

def load_cmu(filename, start_end=True, encoding=True):
    """ Load a file containing CMU pronunciation data. Returns
    word orthographies and ARPA phonetic representations. Both 
    are encoded as a sequence of numbers. """

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
    """ Add start and end tokens to sequence """
    ret = [start] + list(seq)
    ret.append(end)
    return ret

def convert_list(list, map):
    """ Apply an encoding map to each element in list. """
    ret = []
    for word in list:
        ret.append(convert_word(word, map))
    return ret

def convert_word(word, map):
    """ Convert word to encoding using encoding map """
    ret = []
    for ch in word:
        ret.append(map[ch])
    return ret

def load_maps(sym_file):
    """ Return encoding maps generated based on sym_file. """
    syms = load_symbols(sym_file)
    return create_map(syms)

def load_symbols(filename):
    """ Load symbols saved in file """
    return [sym[:-1] for sym in open(filename)]

def create_map(symbols):
    """ Create a symbol-to-encoding map and an
    encoding-to-symbol map. """
    sym_to_code = {sym : i for i, sym in enumerate(symbols)}
    code_to_sym = {i : sym for i, sym in enumerate(symbols)}
    return sym_to_code, code_to_sym


def extend_batch(batch, batch_size):
    """ Fill batch with dummy examples so that it fits batch_size """
    new_batch = {}
    new_batch["X"] = extend_dim_one(batch["X"], batch_size)
    new_batch["Y_in"] = extend_dim_one(batch["Y_in"], batch_size)
    new_batch["Y_targ"] = extend_dim_one(batch["Y_targ"], batch_size)
    new_batch["len_X"] = extend_dim_zero(batch["len_X"], batch_size)
    new_batch["len_Y"] = extend_dim_zero(batch["len_Y"], batch_size)
    return new_batch

def extend_dim_zero(array, new_size):
    x, = array.shape
    ret = np.zeros((new_size))
    ret[:x] = array
    return ret

def extend_dim_one(array, new_size):
    x, y = array.shape
    ret = np.zeros((x, new_size))
    ret[:, :y] = array
    return ret