""" Splits the cmu dictionary file into a train, dev and test set. Also creates
smaller debug and sample files. Writes unique chars and arpa symbols to file """

import random
random.seed(1)

FILENAME = "cmudict-0.7b"
PAD_TOKEN = "<PAD>"
START_TOKEN = "<START>"
END_TOKEN = "<END>"
SPLIT = (0.8, 0.1, 0.1) # train, dev, test split

def shuffle_cmu(filename):
    cmu_dict_file = open(filename, encoding="ISO-8859-1")
    cmu_lines = [l[:-1] for l in cmu_dict_file if l[0].isalpha()]
    random.shuffle(cmu_lines)
    cmu_dict = [l.split(maxsplit=1) for l in cmu_lines]
    return cmu_dict

def write_chars_arpa(cmu_dict):
    chars = set()
    arpa = set()
    for word, pronunciation in cmu_dict:
        for ch in word:
            if ch:
                chars.add(ch)
        phones = pronunciation.split()
        for ph in phones:
            arpa.add(ph)
    with open("characters", "w") as char_file:
        char_file.write(PAD_TOKEN + "\n")
        char_file.write(START_TOKEN + "\n")
        char_file.write(END_TOKEN + "\n")
        for ch in sorted(chars):
            char_file.write(ch + "\n")

    with open("arpabet", "w") as arpa_file:
        arpa_file.write(PAD_TOKEN + "\n")
        arpa_file.write(START_TOKEN + "\n")
        arpa_file.write(END_TOKEN + "\n")
        for ar in sorted(arpa):
            arpa_file.write(ar + "\n")

def train_dev_test_split(data_split):
    assert sum(data_split) == 1.
    vocab_size = len(cmu_dict)
    train = []
    dev = []
    test = []
    for i, entry in enumerate(cmu_dict):
        progress = float(i) / vocab_size
        if progress < data_split[0]:
            train.append(entry)
        elif progress >= data_split[0] and progress < sum(data_split[:2]):
            dev.append(entry)
        else:
            test.append(entry)
    return train, dev, test

def write_data(data, filename):
    with open(filename, "w") as output:
        for word, pronunciation in data:
            output.write(word + "\t" + pronunciation + "\n")

def write_slice(in_filename, out_filename, n):
    lines = [l for l in open(in_filename)][:n]
    with open(out_filename, "w") as out_file:
        for l in lines:
            out_file.write(l)


cmu_dict = shuffle_cmu(FILENAME)
write_chars_arpa(cmu_dict)
train, dev, test = train_dev_test_split(SPLIT)
write_data(train, "train")
write_data(dev, "dev")
write_data(test, "test")
write_slice("train", "train_debug", 4000)
write_slice("dev", "dev_debug", 500)
write_slice("test", "test_debug", 500)
write_slice("train", "sample", 128)
