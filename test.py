import torch
import models
import numpy as np
import os

from PIL import Image
from torchvision import datasets, transforms

cat_to_class = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                '10': 'A', '11': 'B', '12': 'C', '13': 'D', '14': 'E', '15': 'F', '16': 'G',
                '17': 'H', '18': 'J', '19': 'K', '20': 'L', '21': 'M', '22': 'N', '23': 'P',
                '24': 'Q', '25': 'R', '26': 'S', '27': 'T', '28': 'U', '29': 'V', '30': 'W',
                '31': 'X', '32': 'Y', '33': 'Z'}

# Test the data on test set
def test():

    # prepocssing the test data.
    test_transform = transforms.Compose([
            transforms.Resize(28),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()])

    # load the image fron the test set dictionary
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'images','characters', 'test')
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=1)

    # load the pre trained model
    cnn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'cnnnet_win.pkl')
    checkpoint = torch.load(cnn_dir,map_location='cpu')
    model = models.CNN()

    model.eval()

    # load the map relationship of class index to class name
    model.class_to_idx = checkpoint['class_to_idx']
    # load model weights
    model.load_state_dict(checkpoint['state_dict'])

    results = []
    for data in test_loader:

        # run the predition step on cpu
        images, labels = data
        images = images
        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        print(preds)

        for p in np.array(preds.cpu()):
            results.append(cat_to_class[model.class_to_idx[p]])

    print(results)
    return preds
