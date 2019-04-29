
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

# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(device)

file_name_extension = '5000rgbRt' #choose the corresponding database to use

batch_size = 6

n_epochs = 40

target_size = (512, 512)

cubes_file = 'Npydatabase/cubes_{}.npy'.format(file_name_extension)
silhouettes_file = 'Npydatabase/sils_{}.npy'.format(file_name_extension)
parameters_file = 'Npydatabase/params_{}.npy'.format(file_name_extension)

fileExtension = 'TEST'

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

import torch.optim as optim

model = resnet50(cifar=True)
model = model.to(device)  # transfer the neural net onto the GPU
criterion = nn.MSELoss()
# learning_rate = 0.001
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#  ---------------------------------------------------------------
import numpy as np
import tqdm

def train(model, train_dataloader, val_dataloader, n_epochs, loss_function):
    # monitor loss functions as the training progresses
    learning_rate = 0.01
    train_losses = []
    train_epoch_losses = []
    val_losses = []

    val_epoch_losses = []

    best_score  = 1000
    noDecreaseCount = 0

    f = open("results/{}_{}_{}_batchs_{}_epochs_{}_losses.txt".format(date4File, cubeSetName, str(batch_size), str(n_epochs), fileExtension), "w+")
    g = open("results/{}_{}_{}_batchs_{}_epochs_{}_Rtvalues.txt".format(date4File, cubeSetName, str(batch_size), str(n_epochs), fileExtension), "w+")
    g.write('batch angle (error in degree) translation (error in m)  \r\n')
    for epoch in range(n_epochs):

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        f.write('Train, run epoch: {}/{} with Lr {} \r\n'.format(epoch, n_epochs, str(learning_rate)))
        g.write('Train, run epoch: {}/{} with Lr {} \r\n'.format(epoch, n_epochs, str(learning_rate)))
        print('run epoch: {} with Lr {}'.format(epoch, learning_rate))

        ## Training phase
        model.train()
        parameters = []  # ground truth labels
        predict_params = []  # predicted labels

        losses = []  # running loss
        loop = tqdm.tqdm(train_dataloader)
        count = 0

        for image, silhouette, parameter in loop:
            image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
            parameter = parameter.to(device)
            predicted_params = model(image)  # run prediction; output <- vector with probabilities of each class


            # zero the parameter gradients
            optimizer.zero_grad()

            loss = loss_function(predicted_params, parameter) #MSE  value ?
            alpha_loss = loss_function(predicted_params[:, 0], parameter[:, 0])
            beta_loss = loss_function(predicted_params[:, 1], parameter[:, 1])
            gamma_loss = loss_function(predicted_params[:, 2], parameter[:, 2])
            x_loss = loss_function(predicted_params[:, 3], parameter[:, 3])
            y_loss = loss_function(predicted_params[:, 4], parameter[:, 4])
            z_loss = loss_function(predicted_params[:, 5], parameter[:, 5])

            loss.backward()
            optimizer.step()

            parameters.extend(parameter.cpu().numpy())  # append ground truth label
            predict_params.extend(predicted_params.detach().cpu().numpy())  # append computed parameters
            losses.append(loss.item())  # batch length is append every time

            # store value GT(ground truth) and predicted param
            for i in range(0, predicted_params .shape[0]):
                g.write('{} '.format(count))
                for j in range(0, 6):
                    estim = predicted_params[i][j].detach().cpu().numpy()
                    gt = parameter[i][j].detach().cpu().numpy()

                    if j < 3:
                        g.write('{:.4f}°'.format(np.rad2deg(estim - gt)))
                    else:
                        g.write('{:.4f} '.format(estim - gt))
                g.write('\r\n')

            train_loss = np.mean(np.array(losses))

            train_losses.append(train_loss)  # global losses array on the way
            print('run: {}/{} MSE train loss: {:.4f}, angle loss: {:.4f} {:.4f} {:.4f} translation loss: {:.4f} {:.4f} {:.4f} '
                    .format(count, len(loop), train_loss, alpha_loss, beta_loss, gamma_loss, x_loss,y_loss, z_loss))
            f.write('run: {}/{} MSE train loss: {:.4f}, angle loss: {:.4f} {:.4f} {:.4f} translation loss: {:.4f} {:.4f} {:.4f}  \r\n'
                    .format(count, len(loop), train_loss, alpha_loss, beta_loss, gamma_loss, x_loss, y_loss, z_loss))

            count = count + 1


        train_epoch_losses.append(np.mean(np.array(losses))) # global losses array on the way

        count2 = 0
        model.eval()
        f.write('Val, run epoch: {}/{} \r\n'.format(epoch, n_epochs))
        loop = tqdm.tqdm(val_dataloader)
        val_epoch_score = 0 #reset score
        for image, silhouette, parameter in loop:

            image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
            parameter = parameter.to(device)
            predicted_params = model(image)  # run prediction; output <- vector with probabilities of each class

            # zero the parameter gradients
            optimizer.zero_grad()

            # images_1 = renderer(vertices_1, faces_1, textures_1, mode='silhouettes') #create the silhouette with the renderer

            loss = loss_function(predicted_params, parameter) #MSE  value ?

            parameters.extend(parameter.cpu().numpy())  # append ground truth label
            losses.append(loss.item())  # running loss


            av_loss = np.mean(np.array(losses))
            val_epoch_score += av_loss #score of this epoch
            val_losses.append(av_loss)  # append current loss score to global losses array

            print('run: {}/{} MSE val loss: {:.4f}\r\n'.format(count2, len(loop), av_loss))
            count2 = count2 + 1

        val_epoch_losses.append(np.mean(np.array(losses)))  # global losses array on the way
        print('Mean val loss for epoch {} is {}'.format(epoch, val_epoch_score))

        if val_epoch_score < best_score:   #is the validation batch loss better than previous one?
            torch.save(model.state_dict(), 'models/{}_TempModel_Best_train_{}_{}_batchs_epochs_n{}_{}.pth'.format(date4File, cubeSetName, str(batch_size), str(epoch), fileExtension))
            print('parameters saved for epoch {}'.format(epoch))

            noDecreaseCount = 0
            best_score = val_epoch_score
        else:                           #the validation batch loss is not better, increase counter
            noDecreaseCount += 1

        if noDecreaseCount > 5:   #if the validation loss does not deacrease after 5 epochs, lower the learning rate
            learning_rate /= 10
            noDecreaseCount = 0


    f.close()
    g.close()

    return train_epoch_losses, val_epoch_losses


#  ------------------------------------------------------------------

train_losses, val_losses = train(model, train_dataloader, val_dataloader, n_epochs, criterion)

#  ------------------------------------------------------------------

torch.save(model.state_dict(), 'models/{}_FinalModel_train_{}_{}_batchs_{}_epochs_{}.pth'.format(date4File, cubeSetName, str(batch_size), str(n_epochs), fileExtension))
print('parameters saved')

#  ------------------------------------------------------------------
