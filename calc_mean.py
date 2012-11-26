
def calc_mean(list_matrix):
    """Returns the mean of a matrix given as a list of lists.
    """
    total = 0
    cnt = 0
    for lst in list_matrix:
        for number in lst:
            total += number
            cnt += 1
    return (1.*total) / cnt

