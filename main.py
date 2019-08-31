from car_plate_location import *
from char_segmentation import *
import character_classification as cc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import random

def draw_result(img,croped,croped_grey,result,output):
    fig = plt.figure(figsize=(16, 16))

    #draw origin
    ax1 = fig.add_subplot(221)
    ax1.axis('off')
    ax1.set_title("Original Image")
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # cmap='gray'
    ax2_1 = fig.add_subplot(822)
    ax2_1.axis('off')
    ax2_1.set_title("Cropped Plate")
    ax2_1.imshow(cv2.cvtColor(croped, cv2.COLOR_BGR2RGB))

    ax2_2 = fig.add_subplot(826)
    ax2_2.axis('off')
    ax2_2.set_title("Cropped Plate")
    ax2_2.imshow(croped_grey,cmap='gray')

    #cropped numbers
    ax3 = fig.add_subplot(223)
    result = np.concatenate(result, axis=1)
    ax3.axis('off')
    ax3.set_title("Cropped Character Image")
    ax3.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


    ax4 = fig.add_subplot(224)
    ax4.imshow(cv2.cvtColor(plate_loc.img, cv2.COLOR_BGR2RGB))

    rect = patches.Rectangle((target_block[0], target_block[1]), (target_block[2] - target_block[0]),
                             (target_block[3] - target_block[1]), linewidth=1, edgecolor='r', facecolor='none')
    ax4.add_patch(rect)
    s = ""
    for i in output:
        s = s +str(i)

    font1 = {'color':  'white','weight': 'normal','size': 20}

    ax4.text(x=target_block[0], y=target_block[1] -10, s=s, fontdict=font1)
    ax4.set_title("Final Result")
    ax4.axis('off')
    # plt.pause(2)
    plt.waitforbuttonpress()

def random_select():
    image_list = []
    onlyfiles = [f for f in listdir(RAW_IMAGE_DIR) if isfile(join(RAW_IMAGE_DIR, f))]
    for image in onlyfiles:
        image_list.append(image)
    img_index = random.randint(0,len(image_list))

    return image_list[img_index]


if __name__ == '__main__':
    # Perform license plate locating and character segmentation on an image
    # The filename must be a 6-character string (true value)
    # filename = random_select()
    onlyfiles = [f for f in listdir(RAW_IMAGE_DIR) if isfile(join(RAW_IMAGE_DIR, f))]
    for image in onlyfiles:
        # image_list.append(image)
        #img_index = random.randint(0,len(image_list))
        # print(image)
        try:
            plate_loc = CarPlateLocation('images/raw/{}'.format(image))

            inital_block = plate_loc.process()
            target_block = plate_loc.cut_plate(inital_block)
            plate = plate_loc.get_cropped_plate(target_block)
            name = image.split('.')[0]
            # The result is a list of greyscale character images
            result, cropped_img = transform_image(plate, name, output=True, view=False)
            output = cc.run_test(result)
            draw_result(plate_loc.img, plate, cropped_img, result, output)
            # os.system('pause')
        except:
            continue
