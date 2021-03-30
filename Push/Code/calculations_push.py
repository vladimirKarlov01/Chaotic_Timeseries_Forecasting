import numpy as np
from numba import jit
from sklearn.cluster import MeanShift
from itertools import product
from multiprocessing import Pool

import time


class Point:
    def __init__(self, real_value):
        self.real_value = real_value


class VirginPoint(Point):
    def __init__(self, real_value):
        super().__init__(real_value)


class MiddlePoint(Point):
    def __init__(self, real_value, predictions_set, predicted_value):
        super().__init__(real_value)
        self.predictions_set = predictions_set
        self.predicted_value = predicted_value

    def add_prediction(self, prediction):
        self.predictions_set = np.append(self.predictions_set, prediction)


class CompletedPoint(Point):
    def __init__(self, real_value):
        super().__init__(real_value)


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

LORENZ = (np.genfromtxt("lorenz.txt"))  # последние k элементов ряда - тестовая выборка
# train = (np.genfromtxt("lorenz.txt", skip_footer=90000))  # ряд без последних k элементов - тренировочная выборка

TEST_BEGIN = 99900
TEST_END = 100000

CLAWS_MAX_DIST = 9
NUMBER_OF_CLAWS = 4

TRAIN_GAP = 1000
TEST_GAP = 1

MAX_NORM_DELTA = 0.02  # было 0.015
MAX_ABS_ERROR = 0.15  # изначально было 0.05

S = 34  # количество предшедствующих точек ряда, необходимое для прогнозирования точки

K_MAX = 4


# @jit
def reforecast(points, predictions_sets, first_not_absolute, i, k):
    print("\nreforcasting started:")
    for template_number in range(len(templates_by_distances)):
        x, y, z = templates_by_distances[template_number]
        for middle_point in range(first_not_absolute, len(points)):  # middle_point - это индекс в points
            pred_sets_index = middle_point - first_not_absolute

            if z >= pred_sets_index:  # +-1
                continue

            left_part = np.array(
                [points[middle_point - z - y - x - 3],
                 points[middle_point - z - y - 2],
                 points[middle_point - z - 1]]
            )

            if np.isnan(np.sum(left_part)):
                # print("template", template_number, "can't be used")
                continue

            for shifted_template in shifts_for_each_template[template_number]:
                if np.linalg.norm(left_part - shifted_template[:3]) <= MAX_NORM_DELTA:
                    predictions_sets[pred_sets_index].append(shifted_template[3])

    print("sets are changed, repredicting values:")
    for middle_point in range(first_not_absolute, len(points)):
        print("  recalculating point", middle_point, pred_sets_index)
        pred_sets_index = middle_point - first_not_absolute

        if predictions_sets[pred_sets_index]:
            predicted_value = sum(predictions_sets[pred_sets_index]) / len(predictions_sets[pred_sets_index])
        else:
            predicted_value = np.nan

        cur_error = abs(LORENZ[i - k + middle_point - S + 1] - predicted_value)
        if np.isnan(predicted_value) or (cur_error > MAX_ABS_ERROR and middle_point != len(points) - 1):
            points[middle_point] = np.nan
            print("%d-th point is unpredictable, error = %f" % (middle_point, cur_error))
        else:
            points[middle_point] = predicted_value
            print("%d-th point is predictable, predicted_value: %f, error = %f" % (middle_point, predicted_value, cur_error))
        print([len(cur_set) for cur_set in predictions_sets])

    return points


# прогнозирование точки i (index) за k шагов вперед; должна вернуть ошибку и прогнозируемость
def predict(i, k):
    print("cur_point = 0:".upper())
    # last_predicted_index = S  # индекс в points последней точки, в который был получен абсолютный прогноз +-1
    complete_points = [CompletedPoint(i, ) for i in range(LORENZ[i - k - 33: i - k + 1])]  # правая граница не включена => это список из 34 + k точек
    new_points = [VirginPoint(i) for i in range(LORENZ[i - k + 1: i + k + 1])]
    points = complete_points + new_points
    # нулевая итерация
    points = fill(points, predictions_sets, i, k)
    # тут необходимо также добавить одно абсолютное значение

    for cur_point in range(1, k):
        print("\n\ncur_point = ".upper(), cur_point, ":", sep='')
        # print("cur_point: ", cur_point)
        points = reforecast(points, predictions_sets, S + cur_point, i, k)
        points = fill(points, predictions_sets, i, k)
        # print("predictions_sets len:", [len(_) for _ in predictions_sets])

    # import matplotlib.pyplot as plt
    #
    # plt.plot(np.linspace(0, k, k), accessible_points[-k - 1:], color="blue")
    # plt.plot(np.linspace(0, k, k), LORENZ[i - k:i], color='red')
    # plt.show()

    return abs(LORENZ[i] - points[-1]), not np.isnan(points[-1])


def process_for_each_k(k):
    sum_of_abs_errors = 0
    nubmer_of_unpredictable = 0

    for i in range(TEST_BEGIN, TEST_BEGIN + TEST_GAP):  # till TEST_END + 1
        (error, is_predictable) = predict(i, k)
        # print("(error, is_predictable):", (error, is_predictable), '\n')
        if (is_predictable):
            sum_of_abs_errors += error
        else:
            nubmer_of_unpredictable += 1

    if (nubmer_of_unpredictable == TEST_GAP):
        k_RMSE = np.nan
    else:
        k_RMSE = sum_of_abs_errors / (TEST_GAP - nubmer_of_unpredictable)

    print("k =", k, k_RMSE, nubmer_of_unpredictable / TEST_GAP)
    return (k_RMSE, nubmer_of_unpredictable / TEST_GAP)


# Generating templates
templates_by_distances = np.array(list(
    product(range(CLAWS_MAX_DIST + 1), range(CLAWS_MAX_DIST + 1), range(CLAWS_MAX_DIST + 1)))
)

# Training - FIT
shifts_for_each_template = []
for template_number in range(len(templates_by_distances)):
    [x, y, z] = templates_by_distances[template_number]
    cur_claws_indexes = np.array([0, x + 1, x + y + 2, x + y + z + 3])
    tmp = cur_claws_indexes + np.arange(TRAIN_GAP - cur_claws_indexes[3]).reshape(-1, 1)
    shifts_for_each_template.append(LORENZ[tmp])

# file_rmse = open("RMSE.txt", 'w')
# file_percent_of_unpredictable = open("percent_of_unpredictable.txt", 'w')

# works = range(1, K_MAX + 1)

# if __name__ == '__main__':
#     with Pool(processes=4) as pool:
#         res = pool.map(process_for_each_k, works)
#         print(res)

# for work in works:
#     process_for_each_k(work)

predict(99950, 15)

# file_rmse.close()
# file_percent_of_unpredictable.close()
