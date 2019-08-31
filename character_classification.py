import matplotlib.pyplot as plt
import train,test
import models
import os
import torch
import models
import numpy as np

from PIL import Image
from torchvision import datasets, transforms
plt.ion()   # interactive mode

is_train = True
is_save = True
model_path = data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

global models
global images

cat_to_class = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5', '6': '6', '7': '7', '8': '8', '9': '9',
                '10': 'A', '11': 'B', '12': 'C', '13': 'D', '14': 'E', '15': 'F', '16': 'G',
                '17': 'H', '18': 'J', '19': 'K', '20': 'L', '21': 'M', '22': 'N', '23': 'P',
                '24': 'Q', '25': 'R', '26': 'S', '27': 'T', '28': 'U', '29': 'V', '30': 'W',
                '31': 'X', '32': 'Y', '33': 'Z'}

def run_test(images):
    '''
    Predict the characters cropped from the plate license.
    :param images: an 6 * 28 * 35 numpy array representing the 6 digitals an letters
    :return:
    '''
    result = []
    for i in images:
        tensor = torch.tensor([[i]], dtype=torch.float)
        # print(tensor.shape)

        # load pre-trained model
        cnn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'cnnnet_win.pkl')
        checkpoint = torch.load(cnn_dir, map_location='cpu')
        model = models.CNN()

        model.eval()
        model.class_to_idx = checkpoint['class_to_idx']
        model.load_state_dict(checkpoint['state_dict'])

        # predict the characters
        output = model.forward(tensor)
        _, preds = torch.max(output, 1)
        print(preds.item())
        for p in np.array(preds.cpu()):
            result.append(cat_to_class[model.class_to_idx[p]])
        print(preds)
    return result




if __name__ == '__main__':

    if is_train:
        model_ft, criterion, optimizer_ft, exp_lr_scheduler = models.define_model()
        models = train.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=15)
        if is_save:
            torch.save({'state_dict': model_ft.state_dict(),
                        'class_to_idx': model_ft.class_to_idx}
                       , os.path.join(model_path, 'cnnnet_win.pkl'))
    else:
        test.test()

