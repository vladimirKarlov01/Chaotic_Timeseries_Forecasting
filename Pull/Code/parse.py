def is_normal_line(line):
    return not line == "\n"

def parse():
    file = open("Testing_results/21-04-2021-pull-mean/pull/pull.txt")
    lines = file.readlines()
    lines = list(filter(is_normal_line, lines))
    for i in range(len(lines)):
        lines[i] = list(map(float, lines[i].split()[2:]))
    file.close()
    lines.sort()
    # print(lines)

    points = open("Testing_results/21-04-2021-pull-mean/pull/points.txt", "w")
    rmse = open("Testing_results/21-04-2021-pull-mean/pull/RMSE.txt", "w")
    unpred_points_percents = open("Testing_results/21-04-2021-pull-mean/pull/percent_of_unpredictable.txt", "w")
    for line in lines:
        print(line[0], file=points)
        print(line[1], file=rmse)
        print(line[2], file=unpred_points_percents)
    rmse.close()
    unpred_points_percents.close()
    points.close()

    # X = []
    # Y = []
    # for line in lines:
    #     X.append(line[0])
    #     Y.append(line[1])
    # print(X)
    # print(Y)
    # return np.array(X), np.array(Y)

parse()