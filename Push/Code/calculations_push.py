import numpy as np
from numba import jit
from sklearn.cluster import MeanShift
from itertools import product
from multiprocessing import Pool

import time

def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

LORENZ = normalize(np.genfromtxt("Materials/lorenz.txt"))  # последние k элементов ряда - тестовая выборка
train = normalize(np.genfromtxt("Materials/ ", skip_footer=90000))  # ряд без последних k элементов - тренировочная выборка

TEST_BEGIN = 99900
TEST_END = 100000

CLAWS_MAX_DIST = 9
NUMBER_OF_CLAWS = 4

TRAIN_GAP = 1000
TEST_GAP = 1

MAX_NORM_DELTA = 0.01  # было 0.015
MAX_ABS_ERROR = 0.05  # изначально было 0.05

S = 34  # количество предшедствующих точек ряда, необходимое для прогнозирования точки

K_MAX = 4


# @jit
def fill(points, predictions_sets, i, k):
    new_points_range = range(min(10, S + k - len(predictions_sets))) # it's 10 or less

    for new_point in new_points_range:  #new_point - то не индекс, а порядковый номер добавляемой точки
        for template_number in range(len(templates_by_distances)):
            x, y, z = templates_by_distances[template_number]

            if z < new_point:
                continue

            left_part = np.array(
                [points[-z + new_point - y - x - 3],
                 points[-z + new_point - y - 2],
                 points[-z + new_point - 1]]
            )

            if np.isnan(np.sum(left_part)):
                # print("template", template_number, "can't be used")
                continue

            for shifted_template in shifts_for_each_template[template_number]:
                if np.linalg.norm(left_part - shifted_template[:3]) <= MAX_NORM_DELTA:
                    predictions_sets[new_point].append(shifted_template[3])

            if predictions_sets[new_point]:
                predicted_value = sum(predictions_sets[new_point]) / len(predictions_sets[new_point])
            else:
                predicted_value = np.nan

            # ЧТО С НЕПРОГНОЗИРУЕМОСТЬЮ ПОСЛЕДНЕЙ
            # Имеем дело с последней точкой, когда длина point равна максимальной - 1, то есть
            # len(points) == k + S - 1

            if (np.isnan(predicted_value) or (abs(LORENZ[i - k + new_point] - predicted_value) > MAX_ABS_ERROR)):
                points = np.append(points, np.nan)
                # print("%d-th point is unpredictable, error = %f\n" % (cur_point, abs_errors[-1]))
            else:
                points = np.append(points, predicted_value)
                # print("%d-th point is predictable, predicted_value: %f, error = %f" % (cur_point, predicted_value, abs_errors[-1]))

# @jit
def reforecast(points, predictions_sets, last_predicted_index, shifts_for_each_template, i, k):
    # considering all interim points
    for middle_point in range(last_predicted_index, len(points)):  # middle_point - это индекс в points
        for template_number in range(len(templates_by_distances)):
            x, y, z = templates_by_distances[template_number]
            pred_sets_index = middle_point - last_predicted_index

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

        if (np.isnan(predicted_value) or (abs_errors[-1] > MAX_ABS_ERROR and middle_point != steps)): # ЧТО С НЕПРОГНОЗИРУЕМОСТЬЮ ПОСЛЕДНЕЙ
            accessible_points = np.append(points, np.nan)
            # print("%d-th point is unpredictable, error = %f\n" % (cur_point, abs_errors[-1]))
        else:
            accessible_points = np.append(points, predicted_value)
            # print("%d-th point is predictable, predicted_value: %f, error = %f" % (cur_point, predicted_value, abs_errors[-1]))

# прогнозирование точки point_to_forecast (index) за k шагов вперед; должна вернуть ошибку и прогнозируемость
def predict(i, k):
    predictions_sets = np.array([[]] * k)
    # last_predicted_index = S  # индекс в points последней точки, в который был получен абсолютный прогноз +-1
    points = np.array(LORENZ[i - k - 33: i - k + 1])  # правая граница не включена => это список из 34 точек
    abs_errors = np.array([])

    ## нулевая итерация
    fill(points, predictions_sets, shifts_for_each_template, i, k)
    ## тут необходимо также добавить одно абсолютное значение

    for cur_point in range(1, k + 1): # +-1 в правой границе
        # print("cur_point: ", cur_point)
        reforecast(points, predictions_sets, S + cur_point, shifts_for_each_template, k)
        fill(points, predictions_sets, shifts_for_each_template, i, k)
        # print("predictions_sets len:", [len(_) for _ in predictions_sets])
        predicted_value = points[S + cur_point]

        abs_errors = np.append(abs_errors, abs(LORENZ[i - k + cur_point] - predicted_value))

        # prediction = np.array(predictions_list[0]).reshape(-1, 1)
        # if (predictions_list[0].size):
        #     clusters = MeanShift().fit(prediction)
        #     largest_cluster = np.argmax(np.bincount(clusters.labels_))
        #     predicted_value = clusters.cluster_centers_[largest_cluster]
        #     abs_errors = np.append(abs_errors, abs(LORENZ[i - k + cur_point] - predicted_value))
        # else:
        #     abs_errors = np.append(abs_errors, np.nan)



    # import matplotlib.pyplot as plt
    #
    # plt.plot(np.linspace(0, k, k), accessible_points[-k - 1:], color="blue")
    # plt.plot(np.linspace(0, k, k), LORENZ[i - k:i], color='red')
    # plt.show()

    return (abs(LORENZ[i] - points[-1]), not np.isnan(points[-1]))


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
