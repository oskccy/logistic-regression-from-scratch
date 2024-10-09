# logistic regression rounder
def logistic_round(val):
    if 0 <= val <= 1:
        if val >= 0.5:
            return 1
        else:
            return 0
    else:
        print('PREDICTION VALUE NOT IN RANGE [0, 1]')
        return 0
