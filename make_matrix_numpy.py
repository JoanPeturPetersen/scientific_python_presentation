import numpy as np

def make_matrix(N=1000):
    row = np.array(range(1000), dtype=np.int32)
    return np.tile(row, (1000,1))

if __name__ == '__main__':
    make_matrix()
