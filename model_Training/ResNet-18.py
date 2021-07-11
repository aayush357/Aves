

from google.colab import drive
drive.mount('/content/gdrive')

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import seaborn as sns
from torch import tensor

data_transforms={
    'train': transforms.Compose([
      transforms.Resize(256),
      transforms.RandomRotation(45),
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
    'test': transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ]),
}

data_dir='/content/gdrive/MyDrive/images1/'
image_datasets={
    x: datasets.ImageFolder(os.path.join(data_dir,x),data_transforms[x]) for x in ['train','test']
}

dataloaders={x: torch.utils.data.DataLoader(image_datasets[x],batch_size=8,shuffle=True,num_workers=4) for x in ['train','test']
             }

dataset_sizes={x: len(image_datasets[x]) for x in ['train','test']}

class_names=image_datasets['train'].classes

use_gpu=torch.cuda.is_available()

print(dataset_sizes, use_gpu)

def imshow(inp,title=None):
  # converting a tensor to a numpy image
  inp=inp.numpy().transpose((1,2,0))
  mean=np.array([0.485,0.456,0.406])
  std=np.array([0.229, 0.224, 0.225])
  inp=std*inp+mean
  inp=np.clip(inp,0,1)
  plt.imshow(inp)
  if title is not None:
    plt.title(title)
  plt.pause(0.001)

inputs,classes=next(iter(dataloaders['train']))
out=torchvision.utils.make_grid(inputs)
# make_grid produces an image in form of tensors

# imshow(out,title=[class_names[x] for x in classes])

# optimizer are the methods used to change the weights and learning rates to reduce the losses
# epoch means training nn on entire dataset for a forward and backward propagation
# criterion helps in calculating the gradient of the loss
def train_model(model, criterion, optimizer, num_epochs=10):
  since=time.time()
  train_loss,train_acc,test_loss,test_acc=[],[],[],[]
  best_model_wts=model.state_dict()
  best_acc=0.0

  for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch,num_epochs-1))
    print('-'*10)

    for phase in ['train','test']:
      if phase=='train':
        model.train(True)
      else:
        model.train(False)

      running_loss=0.0
      running_corrects=0

      for data in dataloaders[phase]:
        inputs,labels=data
        if use_gpu:
          inputs=Variable(inputs.cuda())
          labels=Variable(labels.cuda())

        else:
          inputs,labels=Variable(inputs),Variable(labels)

        optimizer.zero_grad()# setting every parameter's gradients to 0 for every batch
        outputs=model(inputs)
        
        _, preds=torch.max(outputs.data,1)#finding the maximum of the total outputs
        loss=criterion(outputs,labels)# calculating the loss according to the criterion provided

        if phase=='train':
          loss.backward()# computes the gradient of loss w.r.t. parameter(x)
          optimizer.step()# updates the value of parameter using the gradient
        
        
        # running_loss+=loss[0]
        running_loss+=loss.item()
        running_corrects+=torch.sum(preds==labels.data)

        # print(data)

      epoch_loss=running_loss/dataset_sizes[phase]
      epoch_acc=running_corrects.float()/dataset_sizes[phase]
      if phase=="train":
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

      else:
        test_loss.append(epoch_loss)
        test_acc.append(epoch_acc)

      print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase,epoch_loss,epoch_acc))

      if phase=='test' and epoch_acc>best_acc:
        best_acc=epoch_acc
        best_model_wts=model.state_dict()
        state={'model':model_ft.state_dict(),'optim':optimizer_ft.state_dict()}
        torch.save(state,'/content/gdrive/My Drive/best_resnet1.pth')

    print()
  
  time_elapsed=time.time()-since
  print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60,time_elapsed%60))
  print('Best test Acc: {:4f}'.format(best_acc))

  model.load_state_dict(best_model_wts)
  return model

def visualize_model(model,num_images=8):
  images_so_far=0
  fig=plt.figure()

  for i,data in enumerate(dataloaders['test']):
    inputs,labels=data
    if use_gpu:
      inputs,labels=Variable(inputs.cuda()), Variable(labels.cuda())
    else:
      inputs,labels=Variable(inputs),Variable(labels)
    outputs=model(inputs)
    _,preds=torch.max(outputs.data,1)

    for j in range(inputs.size()[0]):
      images_so_far+=1
      ax=plt.subplot(num_images//2,2,images_so_far)
      ax.axis('off')
      ax.set_title('class: {} predicted: {}'.format(class_names[labels.data[j]],class_names[preds[j]]))
      imshow(inputs.cpu().data[j])
      if images_so_far==num_images:
        return

model_ft=models.resnet18(pretrained=True)

num_ftrs=model_ft.fc.in_features
model_ft.fc=nn.Linear(num_ftrs,200)

if use_gpu:
  model_ft=model_ft.cuda()

criterion=nn.CrossEntropyLoss()

optimizer_ft=optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)

model_ft=train_model(model_ft,criterion,optimizer_ft,num_epochs=75)

"""Loading The saved model"""

checkpoint=torch.load('/content/gdrive/MyDrive/best_resnet.pth',map_location=torch.device('cuda'))

model_ft.load_state_dict(checkpoint['model'])
optimizer_ft.load_state_dict(checkpoint['optim'])

visualize_model(model_ft)

model_ft.eval()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def calc_accuracy(model, data):
    model.eval()
    if use_gpu:
      model.cuda()    
    
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for idx, (inputs, labels) in enumerate(dataloaders[data]):
        if use_gpu:
            inputs, labels = inputs.cuda(), labels.cuda()
        # obtain the outputs from the model
        outputs = model.forward(Variable(inputs))
        prec1, prec5 = accuracy(outputs, Variable(labels), topk=(1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        
    return top1 ,top5
    
top1 ,top5 = calc_accuracy(model_ft, 'test')

print("top1 Average Accuracy:",top1.avg)
print("top5 Average Accuracy:",top5.avg)

def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image
    from PIL import Image
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    if title:
        plt.title(title)
    # PyTorch tensors assume the color channel is first
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, top_num=5):
    # Process image
    img = process_image(image_path)
    
    # Numpy -> Tensor
    image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)

    # Probs
    probs = model.forward(Variable(model_input.cuda()))
    soft_max = torch.nn.Softmax(dim=1)
    probs = soft_max(probs)


    
    top_probs, top_labs = probs.topk(top_num)
    top_probs, top_labs = top_probs.data, top_labs.data
    top_probs = top_probs.cpu().numpy().tolist()[0] 
    top_labs = top_labs.cpu().numpy().tolist()[0]
    print(top_probs)
    
    top_birds = [class_names[lab] for lab in top_labs]
    return top_probs, top_birds

def plot_solution(image_path, model):
    # Set up plot
    plt.figure(figsize = (6,10))
    ax = plt.subplot(2,1,1)

    img = process_image(image_path)
    imshow(img, ax,)
    probs, birds = predict(image_path, model)
     
    # Plot bar chart
    plt.subplot(2,1,2)
    ax=sns.set_style('darkgrid')
    ax=sns.barplot(x=birds, y=probs, color=sns.color_palette()[0])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right')
    plt.xlabel('Bird Names')
    plt.ylabel('Probability')
    plt.show()
image_path = '/content/gdrive/MyDrive/test/054.Blue_Grosbeak/Blue_Grosbeak_0087_36780.jpg'
plot_solution(image_path, model_ft)
