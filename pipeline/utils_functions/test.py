import tqdm
import numpy as np


def testResnet(model, test_dataloader, loss_function, file_name_extension, device, epoch_number=0):
    # monitor loss functions as the training progresses
    test_losses = []

    # test phase
    parameters = []  # ground truth labels
    predicted_params = []
    losses = []  # running loss
    count = 0
    f = open("./results/Test_result_{}_LossRegr_epoch{}.txt".format(file_name_extension, epoch_number), "w+")
    g = open("./results/Test_result_save_param_{}_RtvaluesRegr_epoch{}.txt".format(file_name_extension, epoch_number), "w+")
    g.write('batch angle (error in degree) translation (error in m)  \r\n')

    loop = tqdm.tqdm(test_dataloader)
    for image, silhouette, parameter in loop:

        image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
        parameter = parameter.to(device)
        predicted_param = model(image)  # run prediction; output <- vector with probabilities of each class


        loss = loss_function(predicted_param, parameter) #MSE  value ?

        parameters.extend(parameter.detach().cpu().numpy())  # append ground truth parameters [array([...], dtype=float32), [...], dtype=float32),...)]
        predicted_params.extend(predicted_param.detach().cpu().numpy()) # append computed parameters
        losses.append(loss.item())  # running loss

        alpha_loss = loss_function(predicted_param[:, 0], parameter[:, 0])
        beta_loss = loss_function(predicted_param[:, 1], parameter[:, 1])
        gamma_loss = loss_function(predicted_param[:, 2], parameter[:, 2])
        x_loss = loss_function(predicted_param[:, 3], parameter[:, 3])
        y_loss = loss_function(predicted_param[:, 4], parameter[:, 4])
        z_loss = loss_function(predicted_param[:, 5], parameter[:, 5])

        #store value GT(ground truth) and predicted param
        for i in range(0, predicted_param.shape[0]):
            g.write('{} '.format(count))
            for j in range(0, 6):
                estim = predicted_param[i][j].detach().cpu().numpy()
                gt = parameter[i][j].detach().cpu().numpy()
                if j < 3:
                    g.write('{:.4f}° '.format(np.rad2deg(estim-gt)))
                else:
                    g.write('{:.4f} '.format(estim - gt))
            g.write('\r\n')

        av_loss = np.mean(np.array(losses))
        test_losses.append(av_loss)  # global losses array on the way

        # print('run: {}/{} MSE test loss: {:.4f}\r\n'.format(count, len(loop), av_loss))
        f.write(
            'run: {}/{} MSE train loss: {:.4f}, angle loss: {:.4f} {:.4f} {:.4f} translation loss: {:.4f} {:.4f} {:.4f}  \r\n'
            .format(count, len(loop), av_loss, alpha_loss, beta_loss, gamma_loss, x_loss, y_loss, z_loss))

        count = count + 1

    f.close()
    g.close()

    return test_losses, count, parameters, predicted_params

