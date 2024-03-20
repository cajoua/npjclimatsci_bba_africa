import numpy as np

variable_1 = np.array([82.628, 97.028, 81.564, 77.536, 72.822, 69.220, 63.146, 60.861, 67.944, 71.437])  # Replace with your data
variable_2 = np.array([1.404, 2.592, 4.538, 4.138, 4.433, 4.647, 5.013, 5.512, 4.831, 4.512])  # Replace with your weights
variable_3 = np.array([36, 947, 1239, 1282, 1060, 746, 501, 416, 317, 656])

weighted_average = np.sum(variable_1 * variable_3) / np.sum(variable_3)
print("Weighted Average X1:", weighted_average)

weighted_average2 = np.sum(variable_2 * variable_3) / np.sum(variable_3)
print("Weighted Average X2:", weighted_average2)
