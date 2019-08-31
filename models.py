import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim import lr_scheduler


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1,28,35)
            nn.Conv2d(in_channels=1,  # input height
                      out_channels=24,  # n_filter
                      kernel_size=5,  # filter size
                      stride=1,  # filter step
                      padding=2  # padding the image
                      ),  # output shape (16,28,35)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # downsample

        )
        self.conv2 = nn.Sequential(nn.Conv2d(24, 48, 5, 1, 2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2))
        self.out = nn.Linear(2688, 36)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


def define_model():

    # If gpu is avaliable, we train the model on gpu. Otherwise, training it on cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = CNN()
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return model_ft,criterion,optimizer_ft,exp_lr_scheduler
