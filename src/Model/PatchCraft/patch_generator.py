import cv2
import numpy as np

IMAGE_SIZE = 512  # 512x512
PATCH_SIZE = 32  # 32x32
PATCH_NUM = 512 // PATCH_SIZE  # 16
TOTAL_PATCH = PATCH_NUM ** 2  # 256


def img_to_patches(img) -> tuple:
    """
    Returns 32x32 patches of a resized 512x512 images,
    it returns PATCH_NUM^2 patches on grayscale and PATCH_NUM^2 patches
    on the RGB color scale
    --------------------------------------------------------
    ## parameters:
    - input_path: Accepts input path of the image
    """
    color_img = np.asarray(img.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE)))
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY).astype(np.int32)
    colors, grays = [], []
    for i in range(0, IMAGE_SIZE - PATCH_SIZE, PATCH_SIZE):
        for j in range(0, IMAGE_SIZE - PATCH_SIZE, PATCH_SIZE):
            box = (i, i + PATCH_SIZE, j, j + PATCH_SIZE)
            colors.append(color_img[box[0]:box[1], box[2]:box[3]])
            grays.append(gray_img[box[0]:box[1], box[2]:box[3]])

    return np.asarray(colors), np.asarray(grays)


def get_pixel_var_degree(p: np.array) -> int:
    """
    计算纹理丰富度(相邻像素差异)
    gives pixel variation for a given patch
    ---------------------------------------
    ## parameters:
    - patch: accepts a numpy array format of the patch of an image
    """
    l1 = np.abs(np.diff(p, axis=1)).sum()
    l2 = np.abs(np.diff(p, axis=0)).sum()
    l3 = np.abs(p[:-1, :-1] - p[1:, 1:]).sum()
    l4 = np.abs(p[:-1, 1:] - p[1:, :-1]).sum()

    return l1 + l2 + l3 + l4


def extract_textures(values, patches):
    """
    returns arrays of rich texture and poor texture patches respectively
    --------------------------------------------------------------------
    ## parameters:
    - values: list of values that are pixel variances of each patch
    - color_patches: coloured patches of the target image
    """
    threshold = np.mean(values)  # 平均纹理丰富度(设为阈值)
    rich_idx = (values > threshold)
    poor_idx = ~rich_idx
    rich_patches, rich_values = patches[rich_idx], values[rich_idx]
    poor_patches, poor_values = patches[poor_idx], values[poor_idx]

    return rich_patches, rich_values, poor_patches, poor_values


def get_complete_image(patches: np.array, values: np.array, reversed: bool):
    """
    Develops complete 512x512 image from rich and poor texture patches
    ------------------------------------------------------------------
    ## parameters:
    - patches: Takes a array of rich or poor texture patches
    """
    pad_id = np.random.randint(0, len(patches), max(0, TOTAL_PATCH - len(patches)))
    patches = np.concatenate([patches, patches[pad_id]], axis=0)
    values = np.concatenate([values, values[pad_id]], axis=0)
    sort_indices = np.argsort(values)[:TOTAL_PATCH]
    if reversed:
        sort_indices = sort_indices[::-1]
    patches = patches[sort_indices]
    # (PATCH_NUM^2, 32, 32, 3) -> (PATCH_NUM, PATCH_NUM, 32, 32, 3)
    grid = np.asarray(patches).reshape((PATCH_NUM, PATCH_NUM, PATCH_SIZE, PATCH_SIZE, -1))

    img = grid.transpose([0, 2, 1, 3, 4]).reshape([IMAGE_SIZE, IMAGE_SIZE, -1])
    return img


def smash_reconstruct(img):
    """
    Performs the Smash&Reconstruct part of preprocessing
    reference: [link](https://arxiv.org/abs/2311.12397)
    return rich_texture, poor_texture
    ----------------------------------------------------
    ## parameters:
    - input_path: Accepts input path of the image
    """
    color_patches, gray_patches = img_to_patches(img)

    pixel_var_degree = np.asarray([get_pixel_var_degree(patch) for patch in gray_patches])
    r_patch, r_value, p_patch, p_value = extract_textures(pixel_var_degree, color_patches)

    if len(r_patch) == 0:  # 异常情况(纯色图)
        r_patch, r_value = p_patch[-1:], p_value[-1:]
        p_patch, p_value = p_patch[:-1], p_value[:-1]
    if len(p_patch) == 0:  # 异常情况(纯色图)
        p_patch, p_value = r_patch[-1:], r_value[-1:]
        r_patch, r_value = r_patch[:-1], r_value[:-1]

    rich_texture = get_complete_image(r_patch, r_value, True)
    poor_texture = get_complete_image(p_patch, p_value, False)

    return rich_texture, poor_texture
