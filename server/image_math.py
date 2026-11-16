from PIL import Image
from scipy.signal import convolve2d
import numpy as np

def save_filter_result(image_path: str, filter_f, output_path: str):
    img = Image.open(image_path).convert('L')
    img_matrix = np.array(img)

    res_matrix = filter_f(img_matrix)

    res_img = Image.fromarray(res_matrix)
    res_img.save(output_path)

def laplace_filter(img_matrix: np.ndarray) -> np.ndarray:
    laplace_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])

    proc_matrix = convolve2d(img_matrix, laplace_kernel, mode='same', boundary='symm')
    proc_matrix = np.clip(proc_matrix, 0, 255)

    return proc_matrix.astype('uint8')

def sobbel_detection(img_matrix: np.ndarray) -> np.ndarray:
    sobel_x = np.array([
        [-1,0,1],
        [-2,0,2],
        [-1,0,1]
    ])

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])

    grad_x = convolve2d(img_matrix, sobel_x,mode='same',boundary='symm')
    grad_y = convolve2d(img_matrix, sobel_y, mode='same', boundary='symm')

    proc_matrix = np.sqrt(grad_x**2 + grad_y**2)
    proc_matrix = np.clip(proc_matrix,0,255)

    return proc_matrix.astype('uint8')

def gaussian_blur(img_matrix: np.ndarray) -> np.ndarray:
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ])

    gaussian_kernel = gaussian_kernel / 256.0

    proc_matrix = convolve2d(img_matrix,gaussian_kernel,mode='same',boundary='symm')
    proc_matrix = np.clip(proc_matrix,0,255)

    return  proc_matrix.astype('uint8')


def canny_edge_detection(img_matrix: np.ndarray) -> np.ndarray:
    blurred = gaussian_blur(img_matrix).astype(np.float64)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    grad_x = convolve2d(blurred, sobel_x, mode='same', boundary='symm')
    grad_y = convolve2d(blurred, sobel_y, mode='same', boundary='symm')

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    angle = np.arctan2(grad_y, grad_x) * 180.0 / np.pi
    angle[angle < 0] += 180

    Z = np.zeros(magnitude.shape, dtype=np.float32)
    rows, cols = magnitude.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255
            r = 255

            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            elif (22.5 <= angle[i, j] < 67.5):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            elif (67.5 <= angle[i, j] < 112.5):
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            elif (112.5 <= angle[i, j] < 157.5):
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                Z[i, j] = magnitude[i, j]
            else:
                Z[i, j] = 0

    high_threshold = Z.max() * 0.15
    low_threshold = high_threshold * 0.05

    res = np.zeros(Z.shape, dtype=np.int32)
    weak = np.int32(75)
    strong = np.int32(255)

    strong_i, strong_j = np.where(Z >= high_threshold)
    weak_i, weak_j = np.where((Z <= high_threshold) & (Z >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if res[i, j] == weak:
                if ((res[i + 1, j - 1] == strong) or (res[i + 1, j] == strong) or (res[i + 1, j + 1] == strong) or
                        (res[i, j - 1] == strong) or (res[i, j + 1] == strong) or
                        (res[i - 1, j - 1] == strong) or (res[i - 1, j] == strong) or (res[i - 1, j + 1] == strong)):
                    res[i, j] = strong
                else:
                    res[i, j] = 0

    return res.astype('uint8')


