import numpy as np
from numba import njit, float64, int8, int32, typed
from numba.experimental import jitclass
from sklearn.cluster import MeanShift
from itertools import product
from multiprocessing import Pool
import time


class Point:
    def __init__(self, real_value, predictions_set, predicted_value, is_virgin, is_completed, changed, error=np.inf, motifs=[]):
        self.real_value = real_value
        self.predictions_set = predictions_set
        self.predicted_value = predicted_value
        self.is_virgin = is_virgin
        self.is_completed = is_completed
        self.changed = changed
        self.error = error
        self.motifs = motifs

    def info(self):
        print("{ set size:", self.predictions_set.size, "error:", abs(self.real_value - self.predicted_value), "}", end=' ', flush=True)


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


LORENZ = np.genfromtxt("lorenz.txt")  # последние k элементов ряда - тестовая выборка

TEST_BEGIN = 99900
TEST_END = 100000

CLAWS_MAX_DIST = 9
NUMBER_OF_CLAWS = 4

TRAIN_GAP = 1000
TEST_GAP = 1

MAX_NORM_DELTA = 0.01  # было 0.015
MAX_ABS_ERROR = 0.03  # изначально было 0.05

S = 34  # количество предшедствующих точек ряда, необходимое для прогнозирования точки

K_MAX = 100

DAEMON = 1

INITIAL_ITER_NUM = 2
ITER_EPS = 10**(-5)
max_iter_num = 10


def get_points_from_prepared_prediction(i, k):  # считывается только правая часть (НЕ историческая)
    values = np.genfromtxt("prepared_prediction.txt")
    new_points = []
    for value_num in range(len(values)):
        if np.isnan(values[value_num]):
            new_points.append(Point(LORENZ[i - k + 1 + value_num], np.array([]), np.nan, 1, 0, 1))  # непрогнозируемая
        else:
            new_points.append(Point(LORENZ[i - k + 1 + value_num], np.array([]), values[value_num], 0, 0, 1))  # push дал прогноз

    return new_points

import collections
def print_predictions_set(point):
    print(collections.Counter(point.predictions_set))

# @njit
def reforecast(points):
    print("\nreforcasting started:")
    for template_number in range(len(templates_by_distances)):
        x, y, z = templates_by_distances[template_number]
        for right_point in range(S, len(points)):  # right_point - это индекс в points
            # points indexes in vector
            point0 = points[right_point - z - y - x - 3]
            point1 = points[right_point - z - y - 2]
            point2 = points[right_point - z - 1]
            point3 = points[right_point]
            template_vector = np.array([point0, point1, point2, point3])

            for claw_to_shoot in range(NUMBER_OF_CLAWS):
                # shooting_point_index = right_point - NUMBER_OF_CLAWS + claw_to_shoot + 1
                shooting_point = template_vector[claw_to_shoot]

                if shooting_point.is_completed:
                    continue

                indexes_to_compare = list(filter(lambda _: _ != claw_to_shoot, range(4)))

                # вектор из значений ряда для сравнения по норме
                cur_gunpoints = np.array([tmp.predicted_value for tmp in template_vector[indexes_to_compare]])

                if np.isnan(np.sum(cur_gunpoints)):
                    # print("template", template_number, "can't be used")
                    continue

                if template_vector[indexes_to_compare][0].changed + template_vector[indexes_to_compare][1].changed\
                        + template_vector[indexes_to_compare][2].changed == 0 and\
                        not np.isnan(template_vector[claw_to_shoot].predicted_value):
                    continue

                for shifted_template in shifts_for_each_template[template_number]:
                    if np.linalg.norm(cur_gunpoints - shifted_template[indexes_to_compare]) <= MAX_NORM_DELTA:
                        shooting_point.predictions_set = np.append(
                            shooting_point.predictions_set, shifted_template[claw_to_shoot])
                        X = [right_point - S - 3, right_point - S - 2, right_point - S - 1, right_point - S]
                        shooting_point.motifs.append((X, [cur_tmp_point.predicted_value for cur_tmp_point in template_vector]))
                        shooting_point.is_virgin = False
        # for printed_point_index in range(S, len(points)):
        #     points[printed_point_index].info()
        # print('\n')

    for right_point in range(S, len(points)):
        print("  recalculating point", right_point)
        point_obj = points[right_point]

        if point_obj.predictions_set.size:
            new_value = np.mean(point_obj.predictions_set)
            point_obj.changed = (new_value != point_obj.predicted_value)
            point_obj.predicted_value = new_value
        else:
            point_obj.changed = 0
            continue

        cur_error = abs(point_obj.real_value - point_obj.predicted_value)
        point_obj.error = cur_error

        if DAEMON and cur_error > MAX_ABS_ERROR and right_point != len(points) - 1:
            point_obj.predicted_value = np.nan
            print("%d-th point is unpredictable, error = %f" % (right_point, cur_error))
            print("set: ", end='')
            print_predictions_set(points[right_point])

        else:
            print("%d-th point is predictable, predicted_value: %f, error = %f" % (right_point, points[right_point].predicted_value, cur_error))
            print("set: ", end='')
            print_predictions_set(points[right_point])
    # for printed_point_index in range(S, len(points)):
    #     points[printed_point_index].info()
    print('\n')

    return points


# прогнозирование точки i (index) за k шагов вперед; должна вернуть ошибку и прогнозируемость
def predict(i, k):
    complete_points = [Point(_, np.array([]), _, 0, 1, 1) for _ in LORENZ[i - k - 33: i - k + 1]]  # 34 точки
    # new_points = [Point(_, np.array([]), np.nan, 1, 0, 1) for _ in LORENZ[i - k + 1: i + 1]]  # было до i + 34 точки
    new_points = get_points_from_prepared_prediction(i, k)
    points = complete_points + new_points

    iter_vectors = []  # list of vectors of points to predict for each iteration
    iter_errors = np.array([])  # array: rows = iters, columns = points => cell = error on i-th iter on j-th point

    from copy import deepcopy
    set_1 = []
    set_2 = []
    for iter_num in range(INITIAL_ITER_NUM):
        print("iter_num: ", iter_num)
        points = reforecast(points)
        if iter_num == 0:
            set_1 = deepcopy(points[0].motifs)
        elif iter_num == 1:
            set_2 = deepcopy(points[0].motifs)
        # костыль для изменения changed на 0 у completed точек
        if iter_num == 0:
            for tmp_point in range(S):
                points[tmp_point].changed = 0
        iter_vectors.append(np.array([tmp.predicted_value for tmp in points[S:]]))
        iter_errors = np.concatenate((iter_errors, [tmp.error for tmp in points[S:]]), axis=0)

    import matplotlib.pyplot as plt

    print(len(set_1), set_1[:5])
    print(len(set_2), set_2[:5])
    set_difference = set_2[len(set_1) - 1:]
    indexes_for_sample = np.random.choice(np.arange(len(set_difference)), min((len(set_difference) / 10, 100)), replace=False)
    set_diff_sample = set_difference[indexes_for_sample]
    for motif_tuple in set_diff_sample:
        plt.plot(*motif_tuple)
    iter_num = INITIAL_ITER_NUM
    firstVect = iter_vectors[-1]
    secondVect = iter_vectors[-2]
    norm_of_iters_diff = np.inf
    if (np.isnan(firstVect) == np.isnan(secondVect)).all():
        norm_of_iters_diff = np.linalg.norm(firstVect[~np.isnan(firstVect)] - secondVect[~np.isnan(secondVect)])
    print("iter_num:", iter_num, "| difference: ", norm_of_iters_diff)
    # while norm_of_iters_diff > ITER_EPS and iter_num <= max_iter_num:
    # while iter_num <= max_iter_num:
    #     # print("\n\ncur_point = ".upper(), cur_point, ":", sep='')
    #     print("iter_num:", iter_num, "| difference: ", norm_of_iters_diff)
    #     points = reforecast(points)
    #     iter_vectors.append(np.array([tmp.predicted_value for tmp in points[S:]]))
    #     iter_errors = np.concatenate((iter_errors, [tmp.error for tmp in points[S:]]), axis=0)
    #     firstVect = iter_vectors[-1]
    #     secondVect = iter_vectors[-2]
    #     if (np.isnan(firstVect) == np.isnan(secondVect)).all():
    #         norm_of_iters_diff = np.linalg.norm(firstVect[~np.isnan(firstVect)] - secondVect[~np.isnan(secondVect)])
    #     iter_num += 1

    iter_num = 0
    for iter_vector in iter_vectors:
        print("iter_num:", iter_num, "iter vector: |", iter_vector, '|')
        plt.plot(np.linspace(0, k, k), LORENZ[i - k + 1:i + 1], color='red')  # real values
        plt.scatter(np.linspace(0, k, k), iter_vector, color=(0.0, 0.0, iter_num / 10))
        iter_num += 1
    plt.show()
    print("last norm difference: ", norm_of_iters_diff)

    iter_errors = iter_errors.reshape(iter_num, k)
    for j in range(k):
        plt.plot(np.linspace(0, iter_num, iter_num), iter_errors[:, j], color='purple')
        plt.title("errors for %d-th point" % (j + 1))
        plt.ylabel("abs error")
        plt.xlabel("iteration number")
        plt.show()

    return abs(LORENZ[i] - points[-1].predicted_value), not np.isnan(points[-1].predicted_value)


# Generating templates
templates_by_distances = np.array(list(
    product(range(CLAWS_MAX_DIST + 1), range(CLAWS_MAX_DIST + 1), range(CLAWS_MAX_DIST + 1)))
)

# Training - FIT
shifts_for_each_template = np.array([]).reshape((0, TRAIN_GAP - 3, NUMBER_OF_CLAWS))  # (0, 97, 4)
for template_number in range(len(templates_by_distances)):
    [x, y, z] = templates_by_distances[template_number]
    cur_claws_indexes = np.array([0, x + 1, x + y + 2, x + y + z + 3])
    mask_matrix = cur_claws_indexes + np.arange(TRAIN_GAP - cur_claws_indexes[3]).reshape(-1, 1)  # матрица сдвигов
    # nan values to add at the end
    nan_list = [[np.nan, np.nan, np.nan, np.nan] for _ in range(TRAIN_GAP - (x + y + z + 3), TRAIN_GAP - 3)]
    nan_np_array = np.array(nan_list).reshape(len(nan_list), 4)
    current_template_shifts = np.concatenate(
        [LORENZ[mask_matrix], nan_np_array])  # все свдвиги шаблона данной конфигурации + дополнение
    shifts_for_each_template = np.concatenate(
        [shifts_for_each_template, current_template_shifts.reshape((1, TRAIN_GAP - 3, NUMBER_OF_CLAWS))])

#
# # t1 = time.time()
# for k in range(33, K_MAX + 1, 4):
#     sum_of_abs_errors = 0
#     number_of_unpredictable = 0
#
#     works = [[test_point, k] for test_point in range(TEST_BEGIN, TEST_BEGIN + TEST_GAP)]
#
#     if __name__ == '__main__':
#         with Pool(processes=4) as pool:
#             test_points = pool.starmap(predict, works)
#
#     # test_points = [predict(work[0], work[1]) for work in works]
#
#     for (error, is_predictable) in test_points:  # till TEST_END + 1
#         # print("(error, is_predictable):", (error, is_predictable), '\n')
#         if is_predictable:
#             sum_of_abs_errors += error
#         else:
#             number_of_unpredictable += 1
#
#     if number_of_unpredictable == TEST_GAP:
#         k_RMSE = np.nan
#     else:
#         k_RMSE = sum_of_abs_errors / (TEST_GAP - number_of_unpredictable)
#
#     print("k =", k, k_RMSE, number_of_unpredictable / TEST_GAP, flush=True)
#
# # t2 = time.time()
# # print("time:", t2 - t1)
#

predict(13590, 3)



