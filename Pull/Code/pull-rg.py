import numpy as np
from numba import jit, njit
from sklearn.cluster import MeanShift, DBSCAN
from itertools import product
from multiprocessing import Pool
import time


def normalize(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


LORENZ = np.genfromtxt("lorenz.txt")  # последние k элементов ряда - тестовая выборка

# TEST_BEGIN = 99900
# TEST_END = 100000

TEST_BEGIN = 13000
TEST_END = 13400

CLAWS_MAX_DIST = 9
NUMBER_OF_CLAWS = 4

TRAIN_GAP = 1000
TEST_GAP = 10

MAX_NORM_DELTA = 0.007  # было 0.015
MAX_ABS_ERROR = 0.05  # изначально было 0.05

K_MAX = 100

RG = 1


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
    monotonous_growth_counter = 0
    previous_spread = 0
    for cur_point in range(1, k + 1):
        # print("cur_point: ", cur_point)
        prediction_set = np.array(fill_prediction(points, cur_point))


        # print("prediction_set size:", prediction_set.size)
        cur_spread = max(prediction_set) - min(prediction_set)
        if prediction_set.size:
            if cur_spread > previous_spread:
                monotonous_growth_counter += 1
            else:
                monotonous_growth_counter = 0

            previous_spread = cur_spread

            if RG and monotonous_growth_counter >= 3 and cur_point != k:
                points[34 + cur_point - 1] = np.nan
            else:
                points[34 + cur_point - 1] = prediction_set.mean()
                # cur_error = abs(LORENZ[i - k + cur_point] - predicted_value)
        else:
            # cur_error = np.nan
            points[34 + cur_point - 1] = np.nan
            monotonous_growth_counter = 0


    # import matplotlib.pyplot as plt
    # plt.plot(np.linspace(0, k, k), points[-k:], color="blue")
    # plt.plot(np.linspace(0, k, k), LORENZ[i - k + 1:i + 1], color='red')
    # plt.show()

    return abs(LORENZ[i] - points[-1]), not np.isnan(points[-1])


# t1 = time.time()

# Generating templates
templates_by_distances = np.array(list(
    product(range(CLAWS_MAX_DIST + 1), range(CLAWS_MAX_DIST + 1), range(CLAWS_MAX_DIST + 1)))
)

# Training - FIT
shifts_for_each_template = np.array([]).reshape((0, TRAIN_GAP - 3, NUMBER_OF_CLAWS)) # пустой, но нужных размеров
for template_number in range(len(templates_by_distances)):
    [x, y, z] = templates_by_distances[template_number]
    cur_claws_indexes = np.array([0, x + 1, x + y + 2, x + y + z + 3])
    mask_matrix = cur_claws_indexes + np.arange(TRAIN_GAP - cur_claws_indexes[3]).reshape(-1, 1)  # матрица сдвигов
    mask_matrix += 3000
    # nan values to add at the end
    nan_list = [[np.nan, np.nan, np.nan, np.nan] for _ in range(TRAIN_GAP - (x + y + z + 3), TRAIN_GAP - 3)]
    nan_np_array = np.array(nan_list).reshape(len(nan_list), 4)
    current_template_shifts = np.concatenate([LORENZ[mask_matrix], nan_np_array])  # все свдвиги шаблона данной конфигурации + дополнение
    shifts_for_each_template = np.concatenate([shifts_for_each_template, current_template_shifts.reshape((1, TRAIN_GAP - 3, NUMBER_OF_CLAWS))])


for k in range(1, K_MAX + 1, 8):
    sum_of_abs_errors = 0
    number_of_unpredictable = 0

    works = [[test_point, k] for test_point in range(TEST_BEGIN, TEST_BEGIN + TEST_GAP)]  # till TEST_END + 1
    if __name__ == '__main__':
        with Pool(processes=4) as pool:
            test_points = pool.starmap(predict, works)
            # print(test_points)

        for (error, is_predictable) in test_points:
            # print("(error, is_predictable):", (error, is_predictable), '\n')
            if is_predictable:
                sum_of_abs_errors += error
            else:
                number_of_unpredictable += 1

        if number_of_unpredictable == TEST_GAP:
            k_RMSE = np.nan
        else:
            k_RMSE = sum_of_abs_errors / (TEST_GAP - number_of_unpredictable)

        print("k =", k, k_RMSE, number_of_unpredictable / TEST_GAP, flush=True)

