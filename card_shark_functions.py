import pandas
from sklearn import preprocessing
import statistics
import re
from sklearn import metrics

# calculate the frequency of words in abstracts from a single .JSON


def wordFrequency(df_, exclude, process=False):
    abstractlist = []
    words = {}
    journals = {}
    abstractlen = 0

    for i, row in df_.iterrows():
        if process:
            abstract = row['processed_text']
        else:
            abstract = row['text']

        title = row['title']
        journal = row['journal']

        abstractlist.append(abstract)
        abstractlen += 1

        if journal not in journals:
            journals[journal] = 1
        elif journal in journals:
            journals[journal] += 1

        rgx = re.compile("([\w][\w']*\w)")
        abstractWords = [x for x in re.findall(
            rgx, abstract) if x not in exclude]
        titleWords = re.findall(rgx, title)

        for word in abstractWords:
            if word.lower() in words:
                words[word.lower()] += 1
            elif word.lower() not in words:
                words[word.lower()] = 1

    for word in words:
        words[word] = (words[word])/abstractlen
    for journal in journals:
        journals[journal] = ((journals[journal])/abstractlen) * 10

    return words, journals, abstractlist


# make a scoring matrix based on two single-word frequency populations
def matrixMaker(card, ncbi, filtering=False):
    matrix = {}

    for word in card:
        if word in ncbi:
            matrix[word] = (card[word] - ncbi[word])

    if filtering:
        finalMatrix = {x: matrix[x] for x in matrix if (matrix[x] >= 0.05)}
    elif not filtering:
        finalMatrix = matrix

    return finalMatrix


# make a scoring matrix based on two double-word frequency populations
def doubleMatrixMaker(card, ncbi, exclude):
    doubleMatrix = {}

    def assembly_1():
        for abstract in card:
            # sentences = [x for x in re.split(r' (?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', abstract)]
            sentences = abstract.split(". ")
            for sentence in sentences:
                hold = re.findall(r'\b(\w+)', sentence)
                for thing in hold:
                    if thing not in exclude:
                        for word in hold:
                            if word != thing and word not in exclude:
                                yield [thing.lower(), word.lower()]
    for pair in assembly_1():
        pair = sorted(pair)
        pair = "|".join(str(word) for word in pair).lower()
        if pair not in doubleMatrix:
            doubleMatrix[pair] = 1/len(card)
        elif pair in doubleMatrix:
            doubleMatrix[pair] += 1/len(card)

    def assembly_2():
        for abstract in ncbi:
            # sentences = [x for x in re.split(r' (?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', abstract)]
            sentences = abstract.split(". ")
            # print('c1')
            for sentence in sentences:
                hold = re.findall(r'\b(\w+)', sentence)
                for thing in hold:
                    if thing not in exclude:
                        for word in hold:
                            if word != thing and word not in exclude:
                                # print('c2')
                                yield [thing.lower(), word.lower()]
    for pair in assembly_2():
        pair = sorted(pair)
        pair = "|".join(str(word) for word in pair).lower()
        if pair in doubleMatrix:
            doubleMatrix[pair] -= 1/len(ncbi)
        elif pair not in doubleMatrix:
            pass

    return doubleMatrix


# apply the pre-made matrices to the query abstracts
def bluePill(drug, matrixA, matrixJ, matrixD, exclude, df_, process, only_predictions=False):
    scores = {}

    for i, row in df_.iterrows():
        if process:
            abstract = row['processed_text']
        else:
            abstract = row['text']
        pmid = row['pmid']
        journal = row['journal']
        scores[pmid] = 0

        scores[pmid] += len(re.findall(
            r'[a-zA-Z][a-zA-Z][a-zA-Z][a-zA-Z]?-[0-9][0-9]?[0-9]?', abstract))

        rgx = re.compile("([\w][\w']*\w)")
        abstractWords = [x for x in re.findall(
            rgx, abstract) if x not in exclude]

        terms = {"novel": 5,
                 "characteriz": 3,
                 "clinical": 2,
                 "new": 3,
                 "antibiotic": 1,
                 "resistance": 1,
                 "gene": 1
                 }

        for word in abstractWords:
            if word.lower() in matrixA:
                scores[pmid] += matrixA[word.lower()]
            if word.lower() in terms:
                scores[pmid] += terms[word.lower()]

        if journal in matrixJ:
            scores[pmid] += matrixJ[journal]

        def assembly():
            sentences = re.split(
                r' (?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', abstract)
            for sentence in sentences:
                hold = re.findall(r'\b(\w+)', sentence)
                for thing in hold:
                    if thing not in exclude:
                        for word in hold:
                            if word != thing and word not in exclude:
                                yield [thing.lower(), word.lower()]
        for pair in assembly():
            pair = sorted(pair)
            pair = "|".join(str(word) for word in pair).lower()
            if pair in matrixD:
                if matrixD[pair] > 1:
                    scores[pmid] += (matrixD[pair] * 0.25)
            else:
                pass

    try:
        # print(scores.values)
        # set to calculate the median
        limiter = statistics.median(list(scores.values()))
    except:
        limiter = 10

    if limiter > 10:
        pass
    elif 0 < limiter <= 10:
        limiter = 10

    predictions = [1 if scores[pmid] >= limiter else 0 for pmid in scores]

    probas_ = preprocessing.normalize([[scores[pmid] for pmid in scores]])
    
    if only_predictions:
        return predictions        
    
    Y = list(df_['label'])
    cm = metrics.confusion_matrix(Y, predictions)
    mse = metrics.mean_squared_error(Y, predictions)
    fpr, tpr, thresholds = metrics.roc_curve(Y, probas_[0])
    auc = metrics.auc(fpr, tpr)

    precision, recall, _ = metrics.precision_recall_curve(Y, probas_[0])
    p, r, f1, _ = metrics.precision_recall_fscore_support(
        Y, predictions, average="weighted")

    out = dict(
        fpr=fpr,
        tpr=tpr,
        thresholds=thresholds,
        precision=precision,
        recall=recall,
        cm=cm,
        auc=auc,
        mse=mse,
        p=p,
        r=r,
        f1=f1
    )

    return out, predictions
