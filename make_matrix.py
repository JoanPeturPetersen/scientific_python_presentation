
def make_matrix(N=1000):
    matrix_2d = []
    for i in xrange(N):
        repeat_me = range(N)
        matrix_2d.append(repeat_me)
    return matrix_2d

if __name__ == '__main__':
    make_matrix()
