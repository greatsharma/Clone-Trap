import gc
import numpy as np


def levenshtein_distance(pattern, docs, ignore_case=True) -> dict:
    """Do a fuzzy matching of documents using levenshtein distance in linear space complexity 

    Parameters
    ----------
    pattern : str
        The document which you want to match

    docs : list
        The documents which you want to match with

    ignore_case : bool, optional
        Whether the matching is case sensitive or not (default is True)

    Returns
    -------
    similarity_score : dict
        A dictionary of similarity scores of all documents with pattern in the order passed in docs
    """

    if ignore_case:
        pattern = pattern.lower()

    pattern_len = len(pattern)
    similarity_score = {}
    count = 1

    for doc in docs:

        if ignore_case:
            doc = doc.lower()

        if pattern == doc:
            similarity_score['doc' + str(count)] = 1.0
            count += 1
            continue

        doc_len = len(doc)
        cache = [0] * (pattern_len+1)
        space_penalty = 1

        for i in range(pattern_len+1):
            cache[i] = space_penalty*i

        for i in range(1, doc_len+1):

            temp_store = [0] * (pattern_len+1)
            temp_store[0] = cache[0] + space_penalty

            for j in range(1, pattern_len+1):

                miss_penalty = cache[j-1]

                if pattern[j-1] != doc[i-1]:
                    miss_penalty += 1

                temp_store[j] = min([space_penalty+cache[j],
                                     space_penalty+temp_store[j-1],
                                     miss_penalty])

            cache = temp_store
            del temp_store
            gc.collect

        lev_dist = cache[pattern_len]
        similarity_score['doc' + str(count)] = (pattern_len + doc_len -
                                                lev_dist) / float(pattern_len + doc_len)
        count += 1

    return similarity_score


def get_cloners_table(docs, ignore_case=True):

    n_docs = len(docs)
    c_table = np.ones(shape=(n_docs, n_docs), dtype='float32')

    for i in range(n_docs):
        for j in range(i+1, n_docs):
            d2dscrs = list(levenshtein_distance(docs[i], [docs[j]],
                                                ignore_case).values())
            d2dscrs = [round(scr, 2) for scr in d2dscrs]
            c_table[i][j] = d2dscrs[0]
            c_table[j][i] = d2dscrs[0]

    return c_table


def find_cloners_from_table(c_table, thresh):

    n_docs = c_table.shape[0]

    doc_list = []
    for i in range(n_docs):
        doc_list.append(i)

    cloners = []
    n_clnr = 0

    while doc_list:
        i = doc_list[0]
        cloners.append(set())
        n_clnr += 1

        l_del = []
        for j in range(i, n_docs):
            if c_table[i][j] >= thresh:
                cloners[n_clnr-1].add('doc_'+str(j))
                l_del.append(j)

        if len(l_del) == 1:
            del cloners[n_clnr-1]
            n_clnr -= 1

        for ele in l_del:
            doc_list.remove(ele)

    return cloners


def get_cloners(docs, thresh, ignore_case=True):

    doc_list = []
    for i in range(len(docs)):
        doc_list.append(i)

    cloners = []
    n_clnr = 0

    while doc_list:
        d2dscrs = list(levenshtein_distance(docs[0],
                                            docs[1:], ignore_case).values())

        cloners.append(set())
        n_clnr += 1

        cloners[n_clnr-1].add('doc_'+str(doc_list[0]))
        del doc_list[0], docs[0]

        l_del = []
        for ind, scr in enumerate(d2dscrs):
            if scr >= thresh:
                cloners[n_clnr-1].add('doc_' + str(doc_list[ind]))
                l_del.append(ind)

        if not l_del:
            del cloners[n_clnr-1]
            n_clnr -= 1

        for a, ind in enumerate(l_del):
            del doc_list[ind-a]
            del docs[ind-a]

    return cloners


if __name__ == '__main__':

    pattern = 'this is a test for fuzzy wuzzy match'
    docs = ['a test for fuzzy match', 'test fuzzy matching', 'this is a test for fuzzy wuzzy match', 'fuzz wuzz tester',
            "let's test this program", 'this a test fuzzy wuzzy match', 'this test for fuzzy wuzzy match', 'test fuzz matching',
            'this is test for fuzy wuzy match', 'this is a for fuzzy wuzzy match', 'testing fuzz wuzz match', 'fuzzy wuzzy tester']

    similarity_score = levenshtein_distance(pattern, docs)

    from pprint import pprint
    pprint(similarity_score)
