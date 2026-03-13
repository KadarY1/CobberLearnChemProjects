import numpy as np
from scipy.signal import convolve2d

# 5x5 image (X pattern)
image = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1]
])

# Horizontal edge detector
horizontal_filter = np.array([
    [1,  1,  1],
    [0,  0,  0],
    [-1, -1, -1]
])

# Vertical edge detector
vertical_filter = np.array([
    [1,  0, -1],
    [1,  0, -1],
    [1,  0, -1]
])

# Diagonal edge detector
diagonal_filter = np.array([
    [ 1,  0, -1],
    [ 0,  0,  0],
    [-1,  0,  1]
])

# Apply convolution with padding
horizontal_result = convolve2d(image, horizontal_filter, mode='same', boundary='fill', fillvalue=0)
vertical_result = convolve2d(image, vertical_filter, mode='same', boundary='fill', fillvalue=0)
diagonal_result = convolve2d(image, diagonal_filter, mode='same', boundary='fill', fillvalue=0)

# Print results
print("Original Image:\n", image)
print("\nHorizontal Edge Detection:\n", horizontal_result)
print("\nVertical Edge Detection:\n", vertical_result)
print("\nDiagonal Edge Detection:\n", diagonal_result)