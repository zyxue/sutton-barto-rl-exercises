from scipy import stats


def f1(pos):
    mu = [-0.5, -0.2]
    cov = [[0.4, 0.2], [0.3, 0.6]]
    return stats.multivariate_normal.pdf(pos, mu, cov)


def f2(pos):
    mu = [1, 0.5]
    cov = [[0.4, 0.3], [0.1, 0.5]]
    return stats.multivariate_normal.pdf(pos, mu, cov)


def f3(pos):
    mu = [-0, 0]
    cov = [[0.4, 0.3], [0.1, 0.5]]
    return stats.multivariate_normal.pdf(pos, mu, cov)


def f4(pos):
    mu = [-1, 1.5]
    cov = [[0.4, 0.3], [0.1, 0.5]]
    return stats.multivariate_normal.pdf(pos, mu, cov)


def mixed_gaussian(pos):
    return f1(pos) + f2(pos) - f3(pos) - f4(pos) * 0.5
