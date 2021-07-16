import numpy as np
import io
from PIL import Image
from torch import nn, optim, load, from_numpy, FloatTensor
from torch.autograd import Variable
from torchvision import models
import torch

model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 200)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

checkpoint = torch.load('app/model.pth', map_location=torch.device('cpu'))

model_ft.load_state_dict(checkpoint['model'])
optimizer_ft.load_state_dict(checkpoint['optim'])

model_ft.eval()


def transform_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))

    # Crop
    left_margin = (img.width - 224) / 2
    bottom_margin = (img.height - 224) / 2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin, top_margin))

    # Normalize
    img = np.array(img) / 255
    mean = np.array([0.485, 0.456, 0.406])  # provided mean
    std = np.array([0.229, 0.224, 0.225])  # provided std
    img = (img - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    image_tensor = from_numpy(img).type(FloatTensor)

    # Add batch of size 1 to image
    return image_tensor.unsqueeze(0)


def get_prediction(image_tensor):
    output = model_ft(Variable(image_tensor))
    # print(output)
    index = output.data.cpu().numpy().argmax() + 1
    return index
