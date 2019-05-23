import multiprocessing


d = dict()


def f(param):
    d = param["dict"]
    x = param["data"]
    return d[x] * d[x]


def h(d):
    param = {"data": [1, 2, 3],
             "dict": d}
    pool = multiprocessing.Pool()
    res = pool.map(f, param)
    return res


def ff(d):
    for i in h(d):
        yield i


if __name__ == "__main__":
    d = multiprocessing.Manager().dict()
    d = {1: 1, 2: 2, 3: 3}
    for i in ff(d):
        print(i)
