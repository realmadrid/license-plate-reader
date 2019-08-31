import cv2
import glob
import os
from uuid import uuid4


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', 'output')
RAW_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', 'raw')
CROPPED_IMAGES_PATH = 'images/cropped/*'
TARGET_WIDTH = 28
TARGET_HEIGHT = 35


def read_images(pathname):
    """
    Read the images to a list given a path like 'images/cropped/*'
    :param pathname: file path
    :return: a list of color images and a list of corresponding file names
    """
    images_path = sorted(glob.glob(pathname))
    images = []
    names = []
    for path in images_path:
        images.append(cv2.imread(path, cv2.IMREAD_COLOR))
        name = path[-10:].split('.')[0]
        names.append(name)
    return images, names


def get_image_paths(pathname):
    """
    Get the sorted image paths and the corresponding file names
    :param pathname: file path
    :return: a list of file paths and the corresponding file names
    """
    images_path = sorted(glob.glob(pathname))
    names = []
    for path in images_path:
        name = path[-10:].split('.')[0]
        names.append(name)
    return images_path, names


def binarize_image(image):
    max_val = float(image.max())
    min_val = float(image.min())
    my_thres = max_val - ((max_val - min_val) / 2)
    ret, thresh = cv2.threshold(image, my_thres, 255, cv2.THRESH_BINARY)
    return thresh


def transform_image(image, image_name, output=False, view=False):
    """

    :param image: a license plate image
    :param image_name: filename (true license plate string)
    :param output: True if write the images to output directory
    :param view: True if show the result
    :return: None or 6 segemented characters
    """
    shape = image.shape
    if shape[1] / shape[0] > 3.5:
        cropped_img = image[int(0.05 * shape[0]):int(0.95 * shape[0]), int(0.01 * shape[1]):int(0.99 * shape[1])].copy()
    else:
        cropped_img = image[int(0.12 * shape[0]):int(0.88 * shape[0]), int(0.006 * shape[1]):int(0.994 * shape[1])].copy()
    B, G, R = cv2.split(cropped_img)

    img = binarize_image(G)  # Use green channel for binarization
    img = cv2.GaussianBlur(img, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    img = cv2.medianBlur(img, 5)

    struct1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    struct2 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 8))
    img = cv2.erode(img, struct1, iterations=1)
    img = cv2.dilate(img, struct2, iterations=1)

    img_copy = img.copy()
    contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    chars_x = []

    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        # exclude the whole rectangle
        if x == 0 and y == 0:
            continue
        # the height of a character must be greater than 55% of the image height
        if h < img.shape[0] * 0.55:
            continue
        if hierarchy[0, i, 3] == -1:
            continue
        char = img[y: y + h, x: x + w]
        char = resize_binary_char(char, TARGET_WIDTH, TARGET_HEIGHT)
        chars.append(char)
        chars_x.append(x)
        if view:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    if view:
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    chars = [x for _, x in sorted(zip(chars_x, chars))]
    print('{} characters detected in {}'.format(len(chars), image_name))

    # Correct number of characters
    if len(chars) == 6:
        result = []
        for i, char in enumerate(chars):
            if output:
                path = 'images/output/{}'.format(image_name[i])
                if not os.path.exists(path):
                    os.makedirs(path)
                file_name = os.path.join(path, '{}.bmp'.format(uuid4().hex))
                cv2.imwrite(file_name, char)
            result.append(char)

        return result,img
    else:
        print('Wrong number of characters: {}'.format(image_name))
        return None


def resize_binary_char(char, target_width, target_height):
    """
    Padding the character image to target size.
    :param char: a character image
    :param target_width: the wanted image width
    :param target_height: the wanted image height
    :return: a character image in target size (in black background)
    """
    height, width = char.shape
    res = cv2.resize(char, (int(target_height / height * width), target_height), interpolation=cv2.INTER_CUBIC)
    if res.shape[1] < target_width:
        c = target_width - res.shape[1]
        img = cv2.copyMakeBorder(res, 0, 0, int(c / 2), c - int(c / 2), cv2.BORDER_CONSTANT, value=255)
        img = ~img  # reverse black and white
        return img
    else:
        raise Exception('Target width is less than the character width!')


if __name__ == '__main__':
    images, names = read_images(CROPPED_IMAGES_PATH)

    for i, image in enumerate(images):
        transform_image(image, names[i], output=True)
