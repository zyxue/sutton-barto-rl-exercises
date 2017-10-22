import numpy as np

"""The key (strong assumption) to Naive Bayes is that given y the probabilities
of x=1 for each feature are independent
"""


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
    state = {}
    N = matrix.shape[1]

    ###################
    num_pos = category[category == 1].sum()
    num_neg = matrix.shape[0] - num_pos

    # p(phi|y=1)
    phi_pos = matrix[category == 1, :].sum(axis=0) / num_pos
    # p(phi|y=0)
    phi_neg = matrix[category == 0, :].sum(axis=0) / num_neg
    # p(y=1)
    phi = num_pos / matrix.shape[0]
    ###################
    return phi_pos, phi_neg, phi


def nb_test(matrix, state):
    output = np.zeros(matrix.shape[0])
    ###################
    phi_pos, phi_neg, phi = state

    # REFACTOR: numerically unstable
    # add 1e-16 to avoid np.log(0)
    pos = matrix * phi_pos + (1 - matrix) * (1 - phi_pos) + 1e-16
    p_pos = np.exp(np.sum(np.log(pos), axis=1))

    neg = matrix * phi_neg + (1 - matrix) * (1 - phi_neg) + 1e-16
    p_neg = np.exp(np.sum(np.log(neg), axis=1))

    margin = p_pos * phi + p_neg * (1 - phi)

    p1 = p_pos * phi / margin

    print(p_pos, p_neg)
    output[p1 > 0.5] = 1
    return output


def evaluate(output, label):
    error = (output != label).sum() * 1. / len(output)
    print('Error: %1.4f' % error)


def clean_matrix(matrix):
    # matrixrix may contain values other than {0, 1}
    matrix[matrix > 0.5] = 1
    matrix[matrix <= 0.5] = 0


def main():
    trainMatrix, tokenlist, trainCategory = readMatrix('MATRIX.TRAIN')
    clean_matrix(trainMatrix)
    testMatrix, tokenlist, testCategory = readMatrix('MATRIX.TEST')
    clean_matrix(testMatrix)

    state = nb_train(trainMatrix, trainCategory)
    output = nb_test(testMatrix, state)

    evaluate(output, testCategory)
    return


if __name__ == '__main__':
    main()
