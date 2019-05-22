import numpy as np
import tqdm
import torch
from pipeline.utils_functions.test import testResnet

def train(model, train_dataloader, test_dataloader, n_epochs, loss_function, date4File, cubeSetName, batch_size, fileExtension, device):
    # monitor loss functions as the training progresses
    learning_rate = 0.01
    train_losses = []
    train_epoch_losses = []
    val_losses = []

    val_epoch_losses = []

    best_score  = 1000
    noDecreaseCount = 0

    f = open("./results/Train_{}_{}_{}_batchs_{}_epochs_{}_losses_regressionOnly.txt".format(date4File, cubeSetName, str(batch_size), str(n_epochs), fileExtension), "w+")
    g = open("./results/Train_{}_{}_{}_batchs_{}_epochs_{}_Rtvalues_regressionOnly.txt".format(date4File, cubeSetName, str(batch_size), str(n_epochs), fileExtension), "w+")
    g.write('batch angle (error in degree) translation (error in m)  \r\n')
    for epoch in range(n_epochs):

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        f.write('Regression only Train loss, run epoch: {}/{} with Lr {} \r\n'.format(epoch, n_epochs, str(learning_rate)))
        g.write('Train, run epoch: {}/{} with Lr {} \r\n'.format(epoch, n_epochs, str(learning_rate)))
        print('run epoch: {} with Lr {}'.format(epoch, learning_rate))

        ## Training phase
        model.train()
        parameters = []  # ground truth labels
        predict_params = []  # predicted labels

        losses = []  # running loss
        loop = tqdm.tqdm(train_dataloader)
        count = 0
        print('train phase epoch {}'.format(epoch))
        for image, silhouette, parameter in loop:
            image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
            parameter = parameter.to(device)

            predicted_params = model(image)  # run prediction; output <- vector with probabilities of each class


            # zero the parameter gradients
            optimizer.zero_grad()
            # print(predicted_params.requires_grad)
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

        train_epoch_losses.append(np.mean(np.array(losses)))  # global losses array on the way



        #test phase after the training
        print('test phase epoch {}'.format(epoch))
        model.eval()
        test_losses, count, parameters, predicted_params = testResnet(model, test_dataloader, loss_function,
                                                                      fileExtension, device, epoch_number=epoch)

    f.close()
    g.close()

    return train_epoch_losses, val_epoch_losses
