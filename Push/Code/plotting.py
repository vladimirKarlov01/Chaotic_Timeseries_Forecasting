import numpy as np
import matplotlib.pyplot as plt


# no unpredictable points
points1 = np.genfromtxt("Testing_results/28-04-2021-push-mean/push-no-unpred/points.txt")
rmse1 = np.genfromtxt("Testing_results/28-04-2021-push-mean/push-no-unpred/RMSE.txt")
unpred_points_percent1 = np.genfromtxt("Testing_results/28-04-2021-push-mean/push-no-unpred/percent_of_unpredictable.txt")

# with unpredictable points
points2 = np.genfromtxt("Testing_results/28-04-2021-push-mean/push/points.txt")
rmse2 = np.genfromtxt("Testing_results/28-04-2021-push-mean/push/RMSE.txt")
unpred_points_percent2 = np.genfromtxt("Testing_results/28-04-2021-push-mean/push/percent_of_unpredictable.txt")


fig, (ax1, ax2) = plt.subplots(
    nrows=1, ncols=2,
    figsize=(16, 8)
)

# no unpredictable points
ax1.plot(points1, rmse1, c="orange")
ax1.plot(points2, rmse2, c="blue")
ax1.set_title("RMSE")
ax1.set_xlabel("k")
ax1.set_ylabel("rmse")

# with unpredictable points
ax2.plot(points1, unpred_points_percent1, c="orange")
ax2.plot(points2, unpred_points_percent2, c="blue")
ax2.set_title("% of unpredictable points")
ax2.set_xlabel("k")
ax2.set_ylabel("%")

plt.show()

