from difflib import SequenceMatcher


def evaluate(A, B, end_code=2):
    
    similarity_scores = []
    
    zipped = zip(A, B)
    for a_raw, b_raw in zipped:
        a = clean_end(a_raw, end_code)
        b = clean_end(b_raw, end_code)

        sm = SequenceMatcher(a=a, b=b)
        similarity = sm.ratio()
        similarity_scores.append(similarity)

    return similarity_scores

def clean_end(a, token):
    try:
        idx = list(a).index(token)
        return a[:idx]
    except ValueError:
        return list(a)