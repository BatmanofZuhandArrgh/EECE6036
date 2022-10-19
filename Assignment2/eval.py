def get_stats(pred, truth):
    '''
    pred, truth are lists of 0s,1s
    '''
    # print(pred)
    # print(truth)
    TP = sum([x*y for x,y in zip(pred, truth)])
    TN = sum([(1-x)*(1-y) for x,y in zip(pred, truth)])
    FP = len([0 for x, y in zip(truth, pred) if x == 0 and y ==1])
    FN = len([0 for x, y in zip(truth, pred) if x == 1 and y ==0])

    precision = TP/(TP + FP) if (TP + FP) != 0 else 0
    recall = TP/(TP + FN) if (TP + FN) != 0 else 0
    f1 = 2* (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    # print(TP, TN, FP, FN, precision, recall, f1)
    # print('__________________-')

    return TP, TN, FP, FN, precision, recall, f1

if __name__ == '__main__':
    print(get_stats([1,1,0], [1,0,1]))

