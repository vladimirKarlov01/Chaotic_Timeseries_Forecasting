import numpy as np
from numba import jit
from sklearn.cluster import MeanShift
from itertools import product
from multiprocessing import Pool

import time


class Point:
    def __init__(self, real_value, predictions_set, predicted_value, is_virgin, is_completed):
        self.real_value = real_value
        self.predictions_set = predictions_set
        self.predicted_value = predicted_value
        self.is_virgin = is_virgin
        self.is_completed = is_completed

    def info(self):
        print('{ ', self.predictions_set.size, self.real_value, self.predicted_value, '}', end=' ')


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

LORENZ = (np.genfromtxt("lorenz.txt"))  # последние k элементов ряда - тестовая выборка
# train = (np.genfromtxt("lorenz.txt", skip_footer=90000))  # ряд без последних k элементов - тренировочная выборка

TEST_BEGIN = 99900
TEST_END = 100000

CLAWS_MAX_DIST = 9
NUMBER_OF_CLAWS = 4

TRAIN_GAP = 1000
TEST_GAP = 100

MAX_NORM_DELTA = 0.015  # было 0.015
MAX_ABS_ERROR = 0.05  # изначально было 0.05

S = 34  # количество предшедствующих точек ряда, необходимое для прогнозирования точки

K_MAX = 25

@jit
def reforecast(points, first_not_completed):
    # print("\nreforcasting started:")
    for template_number in range(len(templates_by_distances)):
        x, y, z = templates_by_distances[template_number]
        for middle_point in range(first_not_completed, len(points)):  # middle_point - это индекс в points
            if middle_point + z + 1 >= len(points) or points[middle_point].is_virgin or points[middle_point + z + 1].is_completed:
                continue

            left_part = np.array(
                [points[middle_point - y - x - 2].predicted_value,
                 points[middle_point - y - 1].predicted_value,
                 points[middle_point].predicted_value]
            )

            if np.isnan(np.sum(left_part)):
                # print("template", template_number, "can't be used")
                continue

            for shifted_template in shifts_for_each_template[template_number]:
                if np.linalg.norm(left_part - shifted_template[:3]) <= MAX_NORM_DELTA:
                    points[middle_point + z + 1].predictions_set = np.append(points[middle_point + z + 1].predictions_set, shifted_template[3])
                    points[middle_point + z + 1].is_virgin = False

    for middle_point in range(first_not_completed, len(points)):
        # print("  recalculating point", middle_point, )
        point_obj = points[middle_point]

        if point_obj.predictions_set.size:
            point_obj.predicted_value = sum(point_obj.predictions_set) / len(point_obj.predictions_set)
        else:
            continue

        cur_error = abs(point_obj.real_value - point_obj.predicted_value)

        if np.isnan(point_obj.predicted_value):
            point_obj.predicted_value = np.nan
            # print("%d-th point is unpredictable, error = %f" % (middle_point, cur_error))

        # print("%d-th point is predictable, predicted_value: %f, error = %f" % (middle_point, predicted_value, cur_error))
    # for printed_point_index in range(S, len(points)):
    #     points[printed_point_index].info()
    # print('\n')

    points[first_not_completed].is_completed = True
    return points


# прогнозирование точки i (index) за k шагов вперед; должна вернуть ошибку и прогнозируемость
def predict(i, k):
    # print("cur_point = 0:".upper())
    # last_predicted_index = S  # индекс в points последней точки, в который был получен абсолютный прогноз +-1
    complete_points = [Point(_, np.array([]), _, 0, 1) for _ in LORENZ[i - k - 33: i - k + 1]]  # правая граница не включена => это список из 34 + k точек
    new_points = [Point(_, np.array([]), np.nan, 1, 0) for _ in LORENZ[i - k + 1: i + 1]]
    points = complete_points + new_points
    # нулевая итерация

    reforecast(points, S - 10)

    for cur_point in range(1, k):
        # print("\n\ncur_point = ".upper(), cur_point, ":", sep='')
        # print("cur_point: ", cur_point)
        points = reforecast(points, S + cur_point)

    # import matplotlib.pyplot as plt
    #
    # plt.plot(np.linspace(0, k, k), accessible_points[-k - 1:], color="blue")
    # plt.plot(np.linspace(0, k, k), LORENZ[i - k:i], color='red')
    # plt.show()

    return abs(LORENZ[i] - points[-1].predicted_value), not np.isnan(points[-1].predicted_value)


def process_for_each_k(k):
    sum_of_abs_errors = 0
    number_of_unpredictable = 0

    for i in range(TEST_BEGIN, TEST_BEGIN + TEST_GAP):  # till TEST_END + 1
        (error, is_predictable) = predict(i, k)
        # print("(error, is_predictable):", (error, is_predictable), '\n')
        if (is_predictable):
            sum_of_abs_errors += error
        else:
            number_of_unpredictable += 1

    if number_of_unpredictable == TEST_GAP:
        k_RMSE = np.nan
    else:
        k_RMSE = sum_of_abs_errors / (TEST_GAP - number_of_unpredictable)

    print("k =", k, k_RMSE, number_of_unpredictable / TEST_GAP)
    return (k_RMSE, number_of_unpredictable / TEST_GAP)


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

works = range(1, K_MAX + 1, 4)

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        res = pool.map(process_for_each_k, works)
        print(res)

# for work in works:
#     process_for_each_k(work)

# file_rmse.close()
# file_percent_of_unpredictable.close()
