import numpy as np
from numba import njit, float64, int8, int32, typed
from numba.experimental import jitclass
from sklearn.cluster import MeanShift
from itertools import product
from multiprocessing import Pool
import time

spec = [
    ('real_value', float64),
    ('predictions_set', float64[:]),
    ('predicted_value', float64),
    ('is_virgin', int8),
    ('is_completed', int8),
]


# @jitclass(spec)
class Point:
    def __init__(self, real_value, predictions_set, predicted_value, is_virgin, is_completed):
        self.real_value = real_value
        self.predictions_set = predictions_set
        self.predicted_value = predicted_value
        self.is_virgin = is_virgin
        self.is_completed = is_completed

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

MAX_NORM_DELTA = 0.005  # было 0.015
MAX_ABS_ERROR = 0.08  # изначально было 0.05

S = 34  # количество предшедствующих точек ряда, необходимое для прогнозирования точки

K_MAX = 100

DAEMON = 1


# @njit
def reforecast(points, first_not_completed):
    print("\nreforcasting started:")
    for template_number in range(len(templates_by_distances)):
        x, y, z = templates_by_distances[template_number]
        for right_point in range(first_not_completed, len(points)):  # right_point - это индекс в points
            # points indexes in vector
            point0 = points[right_point - z - y - x - 3].predicted_value
            point1 = points[right_point - z - y - 2].predicted_value
            point2 = points[right_point - z - 1].predicted_value
            point3 = points[right_point].predicted_value

            for_shooting_point_0 = np.array([point1, point2, point3])
            for_shooting_point_1 = np.array([point0, point2, point3])
            for_shooting_point_2 = np.array([point0, point1, point3])
            for_shooting_point_3 = np.array([point0, point1, point2])

            gunpoints_indexes = [for_shooting_point_0, for_shooting_point_1, for_shooting_point_2, for_shooting_point_3]

            for shooting_point in range(NUMBER_OF_CLAWS):
                shooting_point_index = right_point - NUMBER_OF_CLAWS + shooting_point + 1
                if points[shooting_point_index].is_completed:
                    continue

                cur_gunpoints = gunpoints_indexes[shooting_point]

                if np.isnan(np.sum(cur_gunpoints)):
                    # print("template", template_number, "can't be used")
                    continue

                for shifted_template in shifts_for_each_template[template_number]:
                    indexes_to_compare = list(filter(lambda _: _ != shooting_point, range(4)))
                    if np.linalg.norm(cur_gunpoints - shifted_template[indexes_to_compare]) <= MAX_NORM_DELTA:
                        points[shooting_point_index].predictions_set = np.append(
                            points[shooting_point_index].predictions_set, shifted_template[3])
                        points[shooting_point_index].is_virgin = False
        # for printed_point_index in range(S, len(points)):
        #     points[printed_point_index].info()
        # print('\n')

    for right_point in range(first_not_completed, len(points)):
        print("  recalculating point", right_point)
        point_obj = points[right_point]

        if point_obj.predictions_set.size:
            pred_set_sum = 0
            for elem in point_obj.predictions_set:
                pred_set_sum += elem

            point_obj.predicted_value = pred_set_sum / len(point_obj.predictions_set)
        else:
            continue

        cur_error = abs(point_obj.real_value - point_obj.predicted_value)

        if (DAEMON and cur_error > MAX_ABS_ERROR and right_point != len(points) - 1):
            point_obj.predicted_value = np.nan
            print("%d-th point is unpredictable, error = %f" % (right_point, cur_error))
        else:
            print("%d-th point is predictable, predicted_value: %f, error = %f" % (right_point, points[right_point].predicted_value, cur_error))
    for printed_point_index in range(S, len(points)):
        points[printed_point_index].info()
    print('\n')

    return points


# прогнозирование точки i (index) за k шагов вперед; должна вернуть ошибку и прогнозируемость
def predict(i, k):
    complete_points = [Point(_, np.array([]), _, 0, 1) for _ in LORENZ[i - k - 33: i - k + 1]]  # 34 точки
    new_points = [Point(_, np.array([]), np.nan, 1, 0) for _ in LORENZ[i - k + 1: i + 34]]
    # k точек
    points = complete_points + new_points
    for cur_point in range(0, k):
        print("\n\ncur_point = ".upper(), cur_point, ":", sep='')
        # print("cur_point: ", cur_point)
        points = reforecast(points, S )

    import matplotlib.pyplot as plt
    plt.plot(np.linspace(0, k, k), [p.predicted_value for p in points[-k:]], color="blue")
    plt.plot(np.linspace(0, k, k), LORENZ[i - k + 1:i + 1], color='red')
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

predict(13110, 10)
