import numpy as np
import tqdm
import torch

def train(model, train_dataloader, val_dataloader, n_epochs, loss_function, date4File, cubeSetName, batch_size, fileExtension, device):
    # monitor loss functions as the training progresses
    learning_rate = 0.01
    train_losses = []
    train_epoch_losses = []
    val_losses = []

    val_epoch_losses = []

    best_score  = 1000
    noDecreaseCount = 0

    f = open("./results/{}_{}_{}_batchs_{}_epochs_{}_losses_regressionOnly.txt".format(date4File, cubeSetName, str(batch_size), str(n_epochs), fileExtension), "w+")
    g = open("./results/{}_{}_{}_batchs_{}_epochs_{}_Rtvalues_regressionOnly.txt".format(date4File, cubeSetName, str(batch_size), str(n_epochs), fileExtension), "w+")
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
            print(predicted_params.requires_grad)
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
            torch.save(model.state_dict(), './models/{}_TempModel_Best_train_{}_{}_batchs_epochs_n{}_{}.pth'.format(date4File, cubeSetName, str(batch_size), str(epoch), fileExtension))
            print('parameters saved for epoch {}'.format(epoch))

            noDecreaseCount = 0
            best_score = val_epoch_score
        else:                           #the validation batch loss is not better, increase counter
            noDecreaseCount += 1

        if noDecreaseCount == 5:   #if the validation loss does not deacrease after 5 epochs, lower the learning rate
            learning_rate /= 10
            noDecreaseCount = 0

    f.close()
    g.close()

    return train_epoch_losses, val_epoch_losses
