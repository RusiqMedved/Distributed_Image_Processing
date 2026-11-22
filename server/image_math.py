from PIL import Image
from scipy.signal import convolve2d
import numpy as np

def save_filter_result(image_path: str, filter_f, output_path: str):
    img = Image.open(image_path).convert('L')
    img_matrix = np.array(img)

    res_matrix = filter_f(img_matrix)

    res_img = Image.fromarray(res_matrix)
    res_img.save(output_path)

def laplace_filter(img_matrix: np.ndarray, raw_float: bool = False) -> np.ndarray:
    laplace_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float64)

    proc_matrix = convolve2d(img_matrix.astype(np.float64), laplace_kernel, mode='same', boundary='symm')
    if raw_float:
        return proc_matrix.astype(np.float32)

    proc_matrix = np.clip(proc_matrix, 0, 255)
    return proc_matrix.astype(np.uint8)


def sobel_detection(img_matrix: np.ndarray, raw_float: bool = False) -> np.ndarray:
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)

    img_float = img_matrix.astype(np.float64)
    grad_x = convolve2d(img_float, sobel_x, mode='same', boundary='symm')
    grad_y = convolve2d(img_float, sobel_y, mode='same', boundary='symm')

    proc_matrix = np.sqrt(grad_x ** 2 + grad_y ** 2)
    proc_matrix = np.clip(proc_matrix, 0, 255)

    if raw_float:
        return proc_matrix.astype(np.float32)

    proc_matrix = np.clip(proc_matrix, 0, 255)
    return proc_matrix.astype(np.uint8)


# Backwards-compatible alias for existing tests/imports
sobbel_detection = sobel_detection


def gaussian_blur(img_matrix: np.ndarray, raw_float: bool = False) -> np.ndarray:
    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=np.float64) / 256.0

    proc_matrix = convolve2d(img_matrix.astype(np.float64), gaussian_kernel, mode='same', boundary='symm')
    proc_matrix = np.clip(proc_matrix, 0, 255)

    if raw_float:
        return proc_matrix.astype(np.float32)

    proc_matrix = np.clip(proc_matrix, 0, 255)
    return proc_matrix.astype(np.uint8)


def canny_edge_detection(img_matrix: np.ndarray, raw_float: bool = False) -> np.ndarray:
    blurred = gaussian_blur(img_matrix, True).astype(np.float64)

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)

    grad_x = convolve2d(blurred, sobel_x, mode='same', boundary='symm')
    grad_y = convolve2d(blurred, sobel_y, mode='same', boundary='symm')

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    angle = np.arctan2(grad_y, grad_x) * 180.0 / np.pi
    angle[angle < 0] += 180

    Z = np.zeros(magnitude.shape, dtype=np.float32)
    rows, cols = magnitude.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            q = 255.0
            r = 255.0

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
                Z[i, j] = 0.0

    max_val = Z.max()
    if max_val <= 0.0:
        return np.zeros(img_matrix.shape, dtype=np.uint8)

    high_threshold = max_val * 0.15
    low_threshold = high_threshold * 0.05

    res = np.zeros(Z.shape, dtype=np.int32)
    weak = np.int32(75)
    strong = np.int32(255)

    strong_i, strong_j = np.where(Z >= high_threshold)
    weak_i, weak_j = np.where((Z < high_threshold) & (Z >= low_threshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if res[i, j] == weak:
                if (
                    (res[i + 1, j - 1] == strong) or (res[i + 1, j] == strong) or (res[i + 1, j + 1] == strong) or
                    (res[i, j - 1] == strong) or (res[i, j + 1] == strong) or
                    (res[i - 1, j - 1] == strong) or (res[i - 1, j] == strong) or (res[i - 1, j + 1] == strong)
                ):
                    res[i, j] = strong
                else:
                    res[i, j] = 0

    if raw_float:
        return res.astype(np.float32)

    res = np.clip(res, 0, 255)
    return res.astype(np.uint8)

def image_sharpening(img_matrix: np.ndarray, raw_float: bool = False) -> np.ndarray:
    """
    Enhances high-frequency details by applying a direct sharpening kernel.
    Requires margin=1 in chunking.
    """
    sharpen_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ], dtype=np.float64)

    proc_matrix = convolve2d(img_matrix.astype(np.float64), sharpen_kernel, mode='same', boundary='symm')
    proc_matrix = np.clip(proc_matrix, 0, 255)

    if raw_float:
        return proc_matrix.astype(np.float32)

    proc_matrix = np.clip(proc_matrix, 0, 255)
    return proc_matrix.astype(np.uint8)


def laplacian_enhancement(img_matrix: np.ndarray, raw_float: bool = False) -> np.ndarray:
    """
    Enhances edges by calculating the second derivative and subtracting it
    from the original image to boost local contrast.
    Requires margin=1 in chunking.
    """
    laplace_kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=np.float64)

    img_float = img_matrix.astype(np.float64)
    edges = convolve2d(img_float, laplace_kernel, mode='same', boundary='symm')

    proc_matrix = img_float - edges
    proc_matrix = np.clip(proc_matrix, 0, 255)

    if raw_float:
        return proc_matrix.astype(np.float32)

    proc_matrix = np.clip(proc_matrix, 0, 255)
    return proc_matrix.astype(np.uint8)


def emboss_effect(img_matrix: np.ndarray, raw_float: bool = False) -> np.ndarray:
    """
    Highlights diagonal edges and adds a 128 mid-gray offset to create a 3D relief effect.
    Requires margin=1 in chunking.
    """
    emboss_kernel = np.array([
        [-2, -1, 0],
        [-1, 1, 1],
        [0, 1, 2]
    ], dtype=np.float64)

    proc_matrix = convolve2d(img_matrix.astype(np.float64), emboss_kernel, mode='same', boundary='symm')
    proc_matrix = proc_matrix + 128.0
    proc_matrix = np.clip(proc_matrix, 0, 255)

    if raw_float:
        return proc_matrix.astype(np.float32)

    proc_matrix = np.clip(proc_matrix, 0, 255)
    return proc_matrix.astype(np.uint8)


def binarize_and_sketch(img_matrix: np.ndarray, raw_float: bool = False, threshold: int = 60) -> np.ndarray:
    """
    Uses Sobel matrices to find edge magnitudes, thresholds the result to pure black/white,
    and inverts it to create a pencil sketch effect.
    Requires margin=1 in chunking.
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)

    img_float = img_matrix.astype(np.float64)
    grad_x = convolve2d(img_float, sobel_x, mode='same', boundary='symm')
    grad_y = convolve2d(img_float, sobel_y, mode='same', boundary='symm')

    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    sketch_matrix = np.where(magnitude > threshold, 0, 255)

    if raw_float:
        return sketch_matrix.astype(np.float32)

    sketch_matrix = np.clip(sketch_matrix, 0, 255)
    return sketch_matrix.astype(np.uint8)


def harris_corner_detection(img_matrix: np.ndarray, raw_float: bool = False, k: float = 0.04, threshold: float = 1e6) -> np.ndarray:
    """
    Detects structural corners by analyzing the eigenvalues of the gradient covariance matrix.
    Requires margin=3 in chunking (radius 1 for 3x3 Sobel + radius 2 for 5x5 Gaussian window).
    """
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float64)

    img_float = img_matrix.astype(np.float64)
    I_x = convolve2d(img_float, sobel_x, mode='same', boundary='symm')
    I_y = convolve2d(img_float, sobel_y, mode='same', boundary='symm')

    I_xx = I_x ** 2
    I_yy = I_y ** 2
    I_xy = I_x * I_y

    gaussian_kernel = np.array([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
    ], dtype=np.float64) / 256.0

    S_xx = convolve2d(I_xx, gaussian_kernel, mode='same', boundary='symm')
    S_yy = convolve2d(I_yy, gaussian_kernel, mode='same', boundary='symm')
    S_xy = convolve2d(I_xy, gaussian_kernel, mode='same', boundary='symm')

    det_M = (S_xx * S_yy) - (S_xy ** 2)
    trace_M = S_xx + S_yy
    R = det_M - k * (trace_M ** 2)

    if raw_float:
        return R.astype(np.float32)


    corner_matrix = np.copy(img_matrix)
    corner_matrix[R > threshold] = 255

    return corner_matrix.astype(np.uint8)

def original_normalized(img_matrix: np.ndarray, raw_float: bool = False) -> np.ndarray:
    if raw_float:
        return img_matrix.astype(np.float32) / 255.0
    return np.clip(img_matrix, 0, 255).astype(np.uint8)

FILTER_REGISTRY = {
    "original":        {"func": original_normalized,     "radius": 0},
    "laplace":         {"func": laplace_filter,          "radius": 1},
    "sobel":           {"func": sobel_detection,         "radius": 1},
    "sharpen":         {"func": image_sharpening,        "radius": 1},
    "laplace_enhance": {"func": laplacian_enhancement,   "radius": 1},
    "emboss":          {"func": emboss_effect,           "radius": 1},
    "sketch":          {"func": binarize_and_sketch,     "radius": 1},
    "gaussian":        {"func": gaussian_blur,           "radius": 2},
    "harris":          {"func": harris_corner_detection, "radius": 3},
    "canny":           {"func": canny_edge_detection,    "radius": 5},
}