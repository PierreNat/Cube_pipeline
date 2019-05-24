import torch
import torch.nn as nn
import numpy as np
import tqdm
import matplotlib.pyplot as plt

from utils_functions.render1item import render_1_image
from utils_functions.resnet50 import resnet50
from utils_functions.test import testResnet


# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

modelName = '052119_TempModel_Best_train_cubes_10000_t_4_batchs_epochs_n3_TEST__Regr'
# modelName = '042819_TempModel_Best_train_cubes_5000rgbRt_6_batchs_epochs_n39_last' #4 Rt 5000 images
# modelName = '042619_TempModel_Best_train_cubes_10000rgbRt_6_batchs_epochs_n37_2000setRt' #4 Rt 2000 images
# modelName = '042619_TempModel_Best_train_cubes_10000rgbAlphaBeta_6_batchs_epochs_n37_2000set2' #alpha beta rotation

file_name_extension = '10000_t'
# file_name_extension = '5000rgbRt'
# file_name_extension = '2000rgbRt'
# file_name_extension = '10000rgbAlphaBeta'

cubes_file = 'Npydatabase/cubes_{}.npy'.format(file_name_extension)
silhouettes_file = 'Npydatabase/sils_{}.npy'.format(file_name_extension)
parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)

target_size = (512, 512)


cubes = np.load(cubes_file)
sils = np.load(silhouettes_file)
params = np.load(parameters_file)


#  ------------------------------------------------------------------
test_length = 1000
batch_size = 6

test_im = cubes[:test_length]
test_sil = sils[:test_length]
test_param = params[:test_length]

# plt.imshow(test_im[5])
# plt.show()
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

transforms = Compose([ ToTensor(),  normalize])

test_dataset = CubeDataset(test_im, test_sil, test_param, transforms)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


#  ------------------------------------------------------------------


# for image, sil, param in test_dataloader:
#
#     # print(image[2])
#     print(image.size(), param.size()) #torch.Size([batch, 3, 512, 512]) torch.Size([batch, 6])
#     im =2
#     print(param[im])  # parameter in form tensor([2.5508, 0.0000, 0.0000, 0.0000, 0.0000, 5.0000])
#
#     image2show = image[im]  # indexing random  one image
#     print(image2show.size()) #torch.Size([3, 512, 512])
#     plt.imshow((image2show * 0.5 + 0.5).numpy().transpose(1, 2, 0))
#     plt.show()
#     break  # break here just to show 1 batch of data


#  ------------------------------------------------------------------


model = resnet50(cifar=False, modelName=modelName) #train with the saved model from the training script
model = model.to(device)  # transfer the neural net onto the GPU
criterion = nn.MSELoss()

#  ------------------------------------------------------------------

# test the model
epochsValLoss = open(
    "./results/TestProtocol_{}_regressionOnly.txt".format(file_name_extension), "w+")
parameters, predicted_params, test_losses, al, bl, gl, xl, yl, zl  = testResnet(model, test_dataloader, criterion, file_name_extension, device)

#TODO insestead
# test_losses, al, bl, gl, xl, yl, zl = testResnet(model, test_dataloader, loss_function,
#                                                  fileExtension, device, epoch_number=epoch)
#
# all_Test_losses.append(test_losses)
# Test_epoch_losses_alpha.append(al)
# Test_epoch_losses_beta.append(bl)
# Test_epoch_losses_gamma.append(gl)
# Test_epoch_losses_x.append(xl)
# Test_epoch_losses_y.append(yl)
# Test_epoch_losses_z.append(zl)
# epochsValLoss.write(
#     'Validation Loss for epoch {} global {:.4f} angle loss: {:.4f} {:.4f} {:.4f} translation loss: {:.4f} {:.4f} {:.4f}  \r\n'
#     .format(epoch, test_losses, al, bl, gl, xl, yl, zl))

#  ------------------------------------------------------------------
# display computed parameter against ground truth


obj_name = 'rubik_color'

nb_im = 7
# loop = tqdm.tqdm(range(0,nb_im))
for i in range(0,nb_im):

    randIm = i+6 #select a random image
    # print('computed parameter_{}: '.format(i+1))
    # print(predicted_params[randIm])
    # print('ground truth parameter_{}: '.format(i+1))
    # print(params[randIm])
    print('angle and translation MSE loss for {}: '.format(i))
    loss_angle = (predicted_params[randIm][0:3] - params[randIm][0:3])**2
    loss_translation = (predicted_params[randIm][3:6]-params[randIm][3:6])**2
    print(loss_angle, loss_translation)
    # print('error {} degree and {} meter '.format(np.rad2deg(predicted_params[randIm][0:3]-params[randIm][0:3]), predicted_params[randIm][3:6]-params[randIm][3:6]))


    im = render_1_image(obj_name, predicted_params[randIm])  # create the dataset


    plt.subplot(2, nb_im, i+1)
    plt.imshow(test_im[randIm])
    plt.title('Ground truth cube {}'.format(i))

    plt.subplot(2, nb_im, i+1+nb_im)
    plt.imshow(im)
    plt.title('Computed cube {}'.format(i))


plt.show()
print('finish')

