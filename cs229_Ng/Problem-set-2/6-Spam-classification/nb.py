import numpy as np


def readMatrix(file):
    fd = open(file, 'r')
    hdr = fd.readline()
    rows, cols = [int(s) for s in fd.readline().strip().split()]
    tokens = fd.readline().strip().split()
    matrix = np.zeros((rows, cols))
    Y = []
    for i, line in enumerate(fd):
        nums = [int(x) for x in line.strip().split()]
        Y.append(nums[0])
        kv = np.array(nums[1:])
        k = np.cumsum(kv[:-1:2])
        v = kv[1::2]
        matrix[i, k] = v
    return matrix, tokens, np.array(Y)


def nb_train(matrix, category):
    # trainMatrix is now a (numTrainDocs x numTokens) matrix.
    # Each row represents a unique document (email).
    # The j-th column of the row $i$ represents the number of times the j-th
    # token appeared in email $i$. 

    # tokenlist is a long string containing the list of all tokens (words).
    # These tokens are easily known by position in the file TOKENS_LIST

    # trainCategory is a (1 x numTrainDocs) vector containing the true 
    # classifications for the documents just read in. The i-th entry gives the 
    # correct class for the i-th email (which corresponds to the i-th row in 
    # the document word matrix).

    # Spam documents are indicated as class 1, and non-spam as class 0.
    # Note that for the SVM, you would want to convert these to +1 and -1.
    state = {}
    vocabulary_size = matrix.shape[1]
    ###################
    mat1 = matrix[category == 1, :]
    mat0 = matrix[category == 0, :]

    # documentation length, i.e. number of tokens in each document
    mat1_doc_lens = mat1.sum(axis=1)
    # yeq1 means "given y equals 1"
    state['phi_yeq1'] = (mat1.sum(axis=0) + 1) / (np.sum(mat1_doc_lens) + vocabulary_size)

    mat0_doc_lens = mat0.sum(axis=1)
    state['phi_yeq0'] = (mat0.sum(axis=0) + 1) / (np.sum(mat0_doc_lens) + vocabulary_size)

    state['phi'] = mat1.shape[0] / (mat1.shape[0] + mat0.shape[0])
    ###################
    return state


def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    log_phi_yeq1 = np.sum(np.log(state['phi_yeq1']) * matrix, axis=1)
    log_phi_yeq0 = np.sum(np.log(state['phi_yeq0']) * matrix, axis=1)
    phi = state['phi']    

    # see corresponding notebook for equations
    ratio = np.exp(log_phi_yeq0 + np.log(1 - phi) - log_phi_yeq1 - np.log(phi))
    probs = 1 / (1 + ratio)
    output[probs > 0.5] = 1
    ###################
    return output


def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)
    return error


def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('spam_data/MATRIX.TRAIN')
    testMatrix, tokenlist, testCategory = readMatrix('spam_data/MATRIX.TEST')

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return

if __name__ == '__main__':
    main()
