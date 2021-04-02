import numpy as np
import matplotlib.pyplot as plt

# RMSE_no_unpred = np.genfromtxt("tests-after-correcting-reverse/RMSE_no_unpred.txt")
# points_no_unpred = np.genfromtxt("tests-after-correcting-reverse/points_no_unpred.txt")

points = np.genfromtxt("Testing_results/points-with-unpred.txt")
RMSE = np.genfromtxt("Testing_results/RMSE-with-unpred.txt")
percent_of_unpredictable = np.genfromtxt("Testing_results/percent_of_unpredictable.txt")

fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2,
    figsize=(16, 8)
)

ax1.plot(points, RMSE, c="orange")
# ax1.plot(points_no_unpred, RMSE_no_unpred, c="blue")
ax1.set_title("RMSE")
ax1.set_xlabel("k")
ax1.set_ylabel("rmse")

ax2.plot(points, percent_of_unpredictable, c="orange")
ax2.plot(np.linspace(1, 100, 100), [0] * 100, c="blue")
ax2.set_title("% of unpredictable points")
ax2.set_xlabel("k")
ax2.set_ylabel("%")

plt.show()

