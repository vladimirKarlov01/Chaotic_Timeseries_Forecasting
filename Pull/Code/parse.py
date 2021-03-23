import numpy as np

def parse():
    file = open("tests-after-correcting-reverse/no_unpred.txt")
    lines = file.readlines()
    for i in range(len(lines)):
        lines[i] = list(map(float, lines[i].split()[2:]))
    file.close()
    lines.sort()
    # print(lines)

    points = open("tests-after-correcting-reverse/points_no_unpred.txt", "w")
    rmse = open("tests-after-correcting-reverse/RMSE_no_unpred.txt", "w")
    # unpred_points = open("tests-after-correcting-reverse/percent_of_unpredictable.txt", "w")
    for line in lines:
        print(line[0], file=points)
        print(line[1], file=rmse)
        # print(line[2], file=unpred_points)
    rmse.close()
    # unpred_points.close()

    # X = []
    # Y = []
    # for line in lines:
    #     X.append(line[0])
    #     Y.append(line[1])
    # print(X)
    # print(Y)
    # return np.array(X), np.array(Y)

parse()