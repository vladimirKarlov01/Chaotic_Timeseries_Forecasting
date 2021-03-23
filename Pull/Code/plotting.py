import numpy as np
import matplotlib.pyplot as plt

RMSE_no_unpred = np.genfromtxt("tests-after-correcting-reverse/RMSE_no_unpred.txt")
points = np.genfromtxt("tests-after-correcting-reverse/points.txt")
points_no_unpred = np.genfromtxt("tests-after-correcting-reverse/points_no_unpred.txt")
RMSE = np.genfromtxt("tests-after-correcting-reverse/RMSE.txt")
percent_of_unpredictable = np.genfromtxt("tests-after-correcting-reverse/percent_of_unpredictable.txt")

fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2,
    figsize=(16, 8)
)
ax1.scatter(points, RMSE, c="orange")
ax1.scatter(points_no_unpred, RMSE_no_unpred, c="blue")
ax1.set_title("RMSE")
ax1.set_xlabel("k")
ax1.set_ylabel("rmse")

ax2.scatter(points, percent_of_unpredictable, c="orange")
ax2.scatter(np.linspace(1, 50, 50), [0] * 50, c="blue")
ax2.set_title("% of unpredictable points")
ax2.set_xlabel("k")
ax2.set_ylabel("%")

plt.show()

