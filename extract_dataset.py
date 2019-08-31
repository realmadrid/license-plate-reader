import logging
from car_plate_location import *
from char_segmentation import *


logging.getLogger().setLevel(logging.INFO)
CROPPED_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', 'cropped')

# for test
# RAW_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', 'raw_test')
# CROPPED_IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images', 'cropped_test')

if __name__ == '__main__':
    paths, names = get_image_paths(os.path.join(RAW_IMAGE_DIR, '*.jpg'))

    for path, name in zip(paths, names):
        try:
            plate_loc = CarPlateLocation(path)
            inital_block = plate_loc.process()
            target_block = plate_loc.cut_plate(inital_block)
            plate = plate_loc.get_cropped_plate(target_block)
            # cv2.imshow('plate', plate)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(CROPPED_IMAGE_DIR, '{}.bmp'.format(name)), plate)
            logging.info('{}.bmp cropped successfully'.format(name))
        except Exception:
            logging.info('Unexpected exception...')

    logging.info('Plate locating finished.')

    images, names = read_images(CROPPED_IMAGES_PATH)
    # images, names = read_images('images/cropped_test/*')  # for test
    for i, image in enumerate(images):
        transform_image(image, names[i], output=True, view=False)
        # transform_image(image, names[i], output=False, view=True)


