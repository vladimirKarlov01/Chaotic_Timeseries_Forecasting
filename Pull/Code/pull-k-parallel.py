import numpy as np
from numba import jit, njit
from sklearn.cluster import MeanShift, DBSCAN
from itertools import product
from multiprocessing import Pool
import time


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


LORENZ = np.genfromtxt("lorenz.txt")  # последние k элементов ряда - тестовая выборка

TEST_BEGIN = 99900
TEST_END = 100000

CLAWS_MAX_DIST = 9
NUMBER_OF_CLAWS = 4

TRAIN_GAP = 10000
TEST_GAP = 100

MAX_NORM_DELTA = 0.007  # было 0.015
MAX_ABS_ERROR = 0.05  # изначально было 0.05

K_MAX = 100


@njit
def fill_prediction(points, cur_point):
    prediction_set = [np.float64(_) for _ in range(0)]
    for template_number in range(len(templates_by_distances)):
        x, y, z = templates_by_distances[template_number]
        left_part = np.array(
            [points[34 + cur_point - 1 - z - y - x - 3],
             points[34 + cur_point - 1 - z - y - 2],
             points[34 + cur_point - 1 - z - 1]]
        )

        if np.isnan(np.sum(left_part)):
            # print("template", template_number, "can't be used")
            continue

        # print("template", template_number, "is OK")

        for shifted_template in shifts_for_each_template[template_number]:
            if not np.isnan(shifted_template[0]) and np.linalg.norm(left_part - shifted_template[:3]) <= MAX_NORM_DELTA:
                prediction_set.append(shifted_template[3])
    return prediction_set


@njit
def predict(i, k):  # прогнозирование точки i за k шагов вперед; должна вернуть ошибку на i и прогнозируемость
    points = np.append(LORENZ[i - k - 33: i - k + 1],
                       [0 for _ in range(k)])  # правая граница не включена => это список из 34 + k точек

    for cur_point in range(1, k + 1):
        # print("cur_point: ", cur_point)
        # prediction_set = np.array(fill_prediction(points, cur_point)).reshape(-1, 1)
        prediction_set = np.array(fill_prediction(points, cur_point))
        # print("prediction_set size:", prediction_set.size)
        if prediction_set.size:
            # clusters = MeanShift().fit(prediction_set)
            # largest_cluster = np.argmax(np.bincount(clusters.labels_))
            # predicted_value = clusters.cluster_centers_[largest_cluster]
            # cur_error = abs(LORENZ[i - k + cur_point] - predicted_value)

            predicted_value = prediction_set.mean()
            cur_error = abs(LORENZ[i - k + cur_point] - predicted_value)
        else:
            cur_error = np.nan
            predicted_value = np.nan

        # print("predicted_value:", predicted_value)
        # print("cur_error:", cur_error)
        if not prediction_set.size or (cur_error > MAX_ABS_ERROR and cur_point != k):
            points[34 + cur_point - 1] = np.nan
            # print("%d-th point is unpredictable, error = %f\n" % (cur_point, cur_error))
        else:
            points[34 + cur_point - 1] = predicted_value
            # print("%d-th point is predictable, predicted_value: %f, error = %f\n" % (cur_point, predicted_value, cur_error))

    # import matplotlib.pyplot as plt
    # plt.plot(np.linspace(0, k, k), points[-k:], color="blue")
    # plt.plot(np.linspace(0, k, k), LORENZ[i - k + 1:i + 1], color='red')
    # plt.show()

    return abs(LORENZ[i] - predicted_value), not np.isnan(predicted_value)


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
    if nubmer_of_unpredictable == TEST_GAP:
        k_RMSE = np.nan
    else:
        k_RMSE = sum_of_abs_errors / (TEST_GAP - nubmer_of_unpredictable)

    # RMSE = np.append(RMSE, k_RMSE)
    # percent_of_unpredictable = np.append(percent_of_unpredictable, nubmer_of_unpredictable / TEST_GAP)
    # print(k_RMSE, flush=True, file=file_rmse)
    # print(nubmer_of_unpredictable / TEST_GAP, flush=True, file=file_percent_of_unpredictable)
    print("k =", k, k_RMSE, nubmer_of_unpredictable / TEST_GAP, flush=True)
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
shifts_for_each_template = np.array([]).reshape(0, TRAIN_GAP - 3, NUMBER_OF_CLAWS)  # (0, 97, 4)
for template_number in range(len(templates_by_distances)):
    [x, y, z] = templates_by_distances[template_number]
    cur_claws_indexes = np.array([0, x + 1, x + y + 2, x + y + z + 3])
    mask_matrix = cur_claws_indexes + np.arange(TRAIN_GAP - cur_claws_indexes[3]).reshape(-1, 1)  # матрица сдвигов
    # nan values to add at the end
    nan_list = [[np.nan, np.nan, np.nan, np.nan] for _ in range(TRAIN_GAP - (x + y + z + 3), TRAIN_GAP - 3)]
    nan_np_array = np.array(nan_list).reshape(len(nan_list), 4)
    current_template_shifts = np.concatenate([LORENZ[mask_matrix], nan_np_array])  # все свдвиги шаблона данной конфигурации + дополнение
    shifts_for_each_template = np.concatenate([shifts_for_each_template, current_template_shifts.reshape(1, TRAIN_GAP - 3, NUMBER_OF_CLAWS)])


works = range(1, K_MAX + 1, 4)

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        res = pool.map(process_for_each_k, works)
        print(res, flush=True)

