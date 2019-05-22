import tqdm
import numpy as np
import torch
from pipeline.utils_functions.renderBatchItem import renderBatchSil

def testRenderResnet(model, test_dataloader, loss_function, file_name_extension, device , obj_name):
    # monitor loss functions as the training progresses
    test_losses = []

    f = open("./results/Test_result_{}_LossRender.txt".format(file_name_extension), "w+")
    g = open("./results/Test_result_save_param_{}_RtvaluesRender.txt".format(file_name_extension), "w+")
    g.write('batch angle (error in degree) translation (error in m)  \r\n')

    parameters = []  # ground truth labels
    predict_params = []
    losses = []  # running loss

    loop = tqdm.tqdm(test_dataloader)
    count = 0

    for image, silhouette, parameter in loop:
        image = image.to(device)  # we have to send the inputs and targets at every step to the GPU too
        silhouette = silhouette.to(device)
        parameter = parameter.to(device)

        #image has size [batch_length, 3, 512, 512]
        #predicted_param is a tensor with torch.siye[batch, 6]
        predicted_params = model(image)  # run prediction; output <- vector containing  the 6 transformation params


        # object, predicted, ground truth, loss , cuda , and bool for printing logic
        loss = renderBatchSil(obj_name, predicted_params, parameter, loss_function, device)

        parameters.extend(parameter.cpu().numpy())  # append ground truth label
        predict_params.extend(predicted_params.detach().cpu().numpy())  # append computed parameters
        losses.append(loss.item())  # batch length is append every time


        alpha_loss = loss_function(predicted_params[:, 0], parameter[:, 0])
        beta_loss = loss_function(predicted_params[:, 1], parameter[:, 1])
        gamma_loss = loss_function(predicted_params[:, 2], parameter[:, 2])
        x_loss = loss_function(predicted_params[:, 3], parameter[:, 3])
        y_loss = loss_function(predicted_params[:, 4], parameter[:, 4])
        z_loss = loss_function(predicted_params[:, 5], parameter[:, 5])

        # store value GT(ground truth) and predicted param difference
        for i in range(0, predicted_params .shape[0]):
            g.write('{} '.format(count))
            for j in range(0, 6):
                estim = predicted_params[i][j].detach().cpu().numpy()
                gt = parameter[i][j].detach().cpu().numpy()

                if j < 3:
                    g.write('{:.4f}°'.format(np.rad2deg(estim)))
                    # g.write('{:.4f}°'.format(np.rad2deg(estim - gt))) #error in degree
                else:
                    g.write('{:.4f} '.format(estim))
                    # g.write('{:.4f} '.format(estim - gt)) #error in meter
            g.write('\r\n')

        # test loss
        av_loss = np.mean(np.array(losses))  #average loss of the batch
        test_losses.append(av_loss)  # global losses array of all batch average loss

        # print('run: {}/{} MSE test loss: {:.4f}\r\n'.format(count, len(loop), av_loss))
        f.write(
            'run: {}/{} MSE train loss: {:.4f}, angle loss: {:.4f} {:.4f} {:.4f} translation loss: {:.4f} {:.4f} {:.4f}  \r\n'
            .format(count, len(loop), av_loss, alpha_loss, beta_loss, gamma_loss, x_loss, y_loss, z_loss))

        count = count + 1

    f.close()
    g.close()

    return test_losses, count, parameters, predict_params

