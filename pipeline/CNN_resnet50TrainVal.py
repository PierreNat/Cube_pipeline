
"""
script to train a resnet 50 network only with n epoch
Version 4
plot x y z alpha beta gamma error
plot render after each epoch
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from pipeline.utils_functions.resnet50 import resnet50
from pipeline.utils_functions.train_val import train

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

file_name_extension = '3000rgbnew'  # choose the corresponding database to use

batch_size = 6

n_epochs = 3

target_size = (512, 512)

cubes_file = 'Npydatabase/cubes_{}.npy'.format(file_name_extension)
silhouettes_file = 'Npydatabase/sils_{}.npy'.format(file_name_extension)
parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)

fileExtension = 'TEST' #string to ad at the end of the file

cubeSetName = 'cubes_{}'.format(file_name_extension) #used to describe the document name

date4File = '042819' #mmddyy

cubes = np.load(cubes_file)
sils = np.load(silhouettes_file)
params = np.load(parameters_file)

#  ------------------------------------------------------------------
ratio = 0.9  # 90%training 10%validation
split = int(len(cubes)*0.9)
test_length = 1000

train_im = cubes[:split]  # 90% training
train_sil = sils[:split]
train_param = params[:split]

val_im = cubes[split:]  # remaining ratio for validation
val_sil = sils[split:]
val_param = params[split:]

test_im = cubes[:test_length]
test_sil = sils[:test_length]
test_param = params[:test_length]

#  ------------------------------------------------------------------

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize, Lambda

class CubeDataset(Dataset):
    # write your code
    def __init__(self, images, silhouettes, parameters, transform=None):
        self.images = images.astype(np.uint8)  # our image
        self.silhouettes = silhouettes.astype(np.uint8)  # our related parameter
        self.parameters = parameters.astype(np.float32)
        self.transform = transform

    def __getitem__(self, index):
        # Anything could go here, e.g. image loading from file or a different structure
        # must return image and center
        sel_images = self.images[index].astype(np.float32) / 255
        sel_sils = self.silhouettes[index]
        sel_params = self.parameters[index]

        if self.transform is not None:
            sel_images = self.transform(sel_images)
            sel_sils = self.transform(sel_sils)

        return sel_images, sel_images, torch.FloatTensor(sel_params)  # return all parameter in tensor form

    def __len__(self):
        return len(self.images)  # return the length of the dataset

#  ------------------------------------------------------------------



normalize = Normalize(mean=[0.5], std=[0.5])
gray_to_rgb = Lambda(lambda x: x.repeat(3, 1, 1))
transforms = Compose([ToTensor(),  normalize])
train_dataset = CubeDataset(train_im, train_sil, train_param, transforms)
val_dataset = CubeDataset(val_im, val_sil, val_param, transforms)
test_dataset = CubeDataset(test_im, test_sil, test_param, transforms)

#  Note:
#  DataLoader(Dataset,int,bool,int)
#  dataset (Dataset) – dataset from which to load the data.
#  batch_size (int, optional) – how many samples per batch to load (default: 1)
#  shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
#  num_workers = n - how many threads in background for efficient loading

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)


for image, sil, param in train_dataloader:

    # print(image[2])
    print(image.size(), param.size()) #torch.Size([batch, 3, 512, 512]) torch.Size([batch, 6])
    im =2
    print(param[im])  # parameter in form tensor([2.5508, 0.0000, 0.0000, 0.0000, 0.0000, 5.0000])

    image2show = image[im]  # indexing random  one image
    print(image2show.size()) #torch.Size([3, 512, 512])
    plt.imshow((image2show * 0.5 + 0.5).numpy().transpose(1, 2, 0))
    plt.show()
    break  # break here just to show 1 batch of data

#  ------------------------------------------------------------------


model = resnet50(cifar=True)
model = model.to(device)  # transfer the neural net onto the GPU
criterion = nn.MSELoss()

#  ------------------------------------------------------------------

train_losses, val_losses = train(model, train_dataloader, val_dataloader, n_epochs, criterion, date4File, cubeSetName, batch_size, fileExtension, device)

#  ------------------------------------------------------------------

torch.save(model.state_dict(), 'models/{}_FinalModel_train_{}_{}_batchs_{}_epochs_{}.pth'.format(date4File, cubeSetName, str(batch_size), str(n_epochs), fileExtension))
print('parameters saved')

#  ------------------------------------------------------------------
