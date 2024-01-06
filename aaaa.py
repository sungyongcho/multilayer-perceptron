import numpy as np


def calculate_a_from_b(y_true, clipped, outputs, b):
    a = -b * (outputs / (np.log(clipped) + (1 - y_true) * np.log(1 - clipped)))
    return a


# Example data as NumPy arrays
y_true_example = np.array([0.8, 0.7, 0.9])
clipped_example = np.array([0.9, 0.85, 0.88])
outputs_example = np.array([2.0, 1.5, 2.5])

# Assume you have a value for b as a NumPy array
b_example = np.array([0.5, 0.6, 0.7])

# Calculate a from b for NumPy arrays
a_example = calculate_a_from_b(
    y_true_example, clipped_example, outputs_example, b_example
)

# Print the results
print("Example data:")
print(f"y_true = {y_true_example}")
print(f"clipped = {clipped_example}")
print(f"outputs = {outputs_example}")
print(f"b_example = {b_example}")
print(f"a_example calculated from b_example = {a_example}")
