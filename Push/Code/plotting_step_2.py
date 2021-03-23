import numpy as np
import matplotlib.pyplot as plt
from parse import parse

RMSE_no_unpred = np.genfromtxt("test_1_results/no_unpred_final.txt")

RMSE = np.genfromtxt("step_2_test/RMSE.txt")
percent_of_unpredictable = np.genfromtxt("step_2_test/unpred_points.txt")

fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2,
    figsize=(16, 8)
)

# PLOTTING RMSE
ax1.plot(np.linspace(1, RMSE.size * 2 + 1, RMSE.size), RMSE, c="orange")
X, Y = parse()
ax1.plot(X, Y, c="blue")
ax1.set_title("RMSE")
ax1.set_xlabel("k")
ax1.set_ylabel("rmse")

# PLOTTING PERCENT OF UNPREDICTABLE POINTS
ax2.plot(np.linspace(1, percent_of_unpredictable.size * 2 + 1, percent_of_unpredictable.size), percent_of_unpredictable, c="orange")
ax2.plot(np.linspace(1, 100, 50), [0] * 50, c="blue")
ax2.set_title("% of unpredictable points")
ax2.set_xlabel("k")
ax2.set_ylabel("%")

plt.show()

