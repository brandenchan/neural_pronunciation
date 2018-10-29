from difflib import SequenceMatcher
import numpy as np

END_CODE = 2

def dev_stats(similarity_scores):
    """ Return the accuracy metric based on how many words
    the model predicts completely correctly. Return a similarity
    metric that is based on how similar each prediction is to the
    gold standard label. """

    accuracy = similarity_scores.count(1) / len(similarity_scores)
    sim = np.mean(similarity_scores)
    return accuracy, sim

def evaluate(A, B, end_code=END_CODE):
    """ Takes two iterables A and B containing sequences.
    Each sequence is stripped of the end_code token and everything
    that comes after it. Returns elementwise the similarity of the
    sequences in A and B. Expects non time major inputs. """
    
    similarity_scores = []
    
    zipped = zip(A, B)
    for a_raw, b_raw in zipped:
        a = clean_end(a_raw, end_code)
        b = clean_end(b_raw, end_code)

        sm = SequenceMatcher(a=a, b=b)
        similarity = sm.ratio()
        similarity_scores.append(similarity)

    return similarity_scores

def clean_end(a, end_token):
    """ Strips sequence a of end_token and everything
    that comes after. """
    try:
        idx = list(a).index(end_token)
        return a[:idx]
    except ValueError:
        return list(a)