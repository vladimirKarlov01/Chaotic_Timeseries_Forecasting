import numpy as np
from numba import jit, njit
from sklearn.cluster import MeanShift
from itertools import product
from multiprocessing import Pool

import time

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

LORENZ = normalize(np.genfromtxt("lorenz.txt"))  # последние k элементов ряда - тестовая выборка
train = np.genfromtxt("lorenz.txt", skip_footer=90000)  # ряд без последних k элементов - тренировочная выборка

TEST_BEGIN = 99900
TEST_END = 100000

CLAWS_MAX_DIST = 9
NUMBER_OF_CLAWS = 4

TRAIN_GAP = 1000
TEST_GAP = 1

MAX_NORM_DELTA = 0.015  # было 0.015
MAX_ABS_ERROR = 0.05  # изначально было 0.05

K_MAX = 50

@jit
def fill_prediction(accessible_points, shifts_for_each_template):
    prediction = np.array([])
    for template_number in range(len(templates_by_distances)):
        (x, y, z) = templates_by_distances[template_number]
        to_forecast = np.array(
            [accessible_points[-z - y - x - 3], accessible_points[-z - y - 2], accessible_points[-z - 1]]
        )

        if np.isnan(np.sum(to_forecast)):
            # print("template", template_number, "can't be used")
            continue

        # print("template", template_number, "is OK")

        for shifted_template in shifts_for_each_template[template_number]:
            if np.linalg.norm(to_forecast - shifted_template[:3]) <= MAX_NORM_DELTA:
                prediction = np.append(prediction, shifted_template[3])
        # print(prediction.size)
    return prediction

def predict(i, k):  # прогнозирование точки i за k шагов вперед; должна вернуть ошибку на i и прогнозируемость
    accessible_points = np.array(LORENZ[i - k - 33: i - k + 1])  # правая граница не включена
    abs_errors = np.array([])

    for cur_point in range(1, k + 1):
        # print("cur_point: ", cur_point)
        prediction = fill_prediction(accessible_points, shifts_for_each_template)
        # print("prediction size:", prediction.size)
        prediction = prediction.reshape(-1, 1)
        if (prediction.size):
            clusters = MeanShift().fit(prediction)
            largest_cluster = np.argmax(np.bincount(clusters.labels_))
            predicted_value = clusters.cluster_centers_[largest_cluster]
            abs_errors = np.append(abs_errors, abs(LORENZ[i - k + cur_point] - predicted_value))
        else:
            abs_errors = np.append(abs_errors, np.nan)


        if (not prediction.size or (abs_errors[-1] > MAX_ABS_ERROR and cur_point != k)):
            accessible_points = np.append(accessible_points, np.nan)
            # print("%d-th point is unpredictable, error = %f\n" % (cur_point, abs_errors[-1]))
        else:
            accessible_points = np.append(accessible_points, predicted_value)
            # print("%d-th point is predictable, predicted_value: %f, error = %f" % (cur_point, predicted_value, abs_errors[-1]))

    return (abs(LORENZ[i] - accessible_points[-1]), not np.isnan(accessible_points[-1]))


def process_for_each_k(k):
    # print("k =", k, "START\n")
    sum_of_abs_errors = 0
    nubmer_of_unpredictable = 0

    for i in range(TEST_BEGIN, TEST_BEGIN + TEST_GAP):  # till TEST_END + 1

        (error, is_predictable) = predict(i, k)
        # print("(error, is_predictable):", (error, is_predictable), '\n')
        if (is_predictable):
            sum_of_abs_errors += error
        else:
            nubmer_of_unpredictable += 1

    # считаем k_RMSE
    # делить на кол-во всех точек или на кол-во прогнозируемых
    if (nubmer_of_unpredictable == TEST_GAP):
        k_RMSE = np.nan
    else:
        k_RMSE = sum_of_abs_errors / (TEST_GAP - nubmer_of_unpredictable)

    # RMSE = np.append(RMSE, k_RMSE)
    # percent_of_unpredictable = np.append(percent_of_unpredictable, nubmer_of_unpredictable / TEST_GAP)
    # print(k_RMSE, flush=True, file=file_rmse)
    # print(nubmer_of_unpredictable / TEST_GAP, flush=True, file=file_percent_of_unpredictable)
    print("k =", k, k_RMSE, nubmer_of_unpredictable / TEST_GAP)
    return (k_RMSE, nubmer_of_unpredictable / TEST_GAP)
    # print("sum_of_abs_errors:", sum_of_abs_errors)
    # print("nubmer_of_unpredictable:", nubmer_of_unpredictable, '\n')
    # print("k =", k, "FINISH\n\n\n")


# t1 = time.time()

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

# Predict + Draw
# RMSE = np.array([])
# percent_of_unpredictable = np.array([])


# file_rmse = open("RMSE.txt", 'w')
# file_percent_of_unpredictable = open("percent_of_unpredictable.txt", 'w')

works = range(1, K_MAX + 1, 2)

# if __name__ == '__main__':
#     with Pool(processes=4) as pool:
#         res = pool.map(process_for_each_k, works)
#         print(res)

for work in works:
    process_for_each_k(work)

t2 = time.time()

# print("TIME:", t2 - t1)
# file_rmse.close()
# file_percent_of_unpredictable.close()
