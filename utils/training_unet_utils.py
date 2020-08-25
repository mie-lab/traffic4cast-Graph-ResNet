import json
import os
import time
import warnings

import torch
from torch.optim.lr_scheduler import StepLR
from utils.earlystopping import EarlyStopping

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from utils.visual_TB import Visualizer


def trainNet(model, train_loader, val_loader, val_loader_ttimes, device, log_dir, config, mask=None):
    # Print all of the hyper parameters of the training iteration:

    # define the optimizer & learning rate
    optim = torch.optim.SGD(model.parameters(), **config['optimizer'])

    scheduler = StepLR(optim, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

    writer = Visualizer(log_dir)

    # dump config file
    with open(os.path.join(log_dir, 'config.json'), 'w') as fp:
        json.dump(config, fp)

    # Time for printing
    training_start_time = time.time()
    globaliter = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config['patience'], verbose=True)

    # Loop for n_epochs
    for epoch_idx, epoch in enumerate(range(config['num_epochs'])):
        writer.write_lr(optim, epoch)

        # train for one epoch
        globaliter = train(model=model, train_loader=train_loader, optim=optim, device=device, writer=writer,
                           epoch=epoch,
                           globaliter=globaliter, config=config, mask=mask)

        # At the end of the epoch, do a pass on the validation set
        val_loss = validate(model=model, val_loader=val_loader, device=device, writer=writer, globaliter=globaliter,
                            mask=mask, config=config)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if config['debug'] and epoch_idx >= 0:
            break

        scheduler.step()

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    model.load_state_dict(torch.load(log_dir + '/checkpoint.pt'))

    # remember to close
    writer.close()


def train(model, train_loader, optim, device, writer, epoch, globaliter, config, mask=None):
    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)

    # define start time
    start_time = time.time()

    if config['model']['depth'] <= 5:
        padd = torch.nn.ZeroPad2d((6, 6, 1, 0))
    elif config['model']['depth'] <= 7:
        padd = torch.nn.ZeroPad2d((6, 6, 8, 9))
    elif config['model']['depth'] == 8:
        padd = torch.nn.ZeroPad2d((38, 38, 8, 9))
    else:
        raise exception("wrong padding for UNET depth")

    mask_padd = padd(mask)

    for i, data in enumerate(train_loader, 0):
        inputs, Y, feature_dict = data
        inputs = inputs / 255
        globaliter = globaliter + 1

        # padd the input data with 0 to ensure same size after upscaling by the network
        # inputs [495, 436] -> [496, 448] resp. [495, 436] -> [512, 448]

        inputs = padd(inputs)
        inputs = inputs.float().to(device)

        # mask input
        masks_input = mask_padd.expand(inputs.shape).float()
        inputs = inputs * masks_input

        # the Y remains the same dimension
        Y = Y.float().to(device)
        masks_output = mask.expand(Y.shape).float()

        # Set the parameter gradients to zero
        optim.zero_grad()

        # Forward pass, backward pass, optimize

        prediction = model(inputs)

        # crop the output for comparing with true Y
        if config['model']['depth'] <= 5:
            loss_size = torch.nn.functional.mse_loss(prediction[:, :, 1:, 6:-6] * masks_output, Y * masks_output)
        elif config['model']['depth'] <= 7:
            loss_size = torch.nn.functional.mse_loss(prediction[:, :, 8:-9, 6:-6] * masks_output, Y * masks_output)
        elif config['model']['depth'] == 8:
            loss_size = torch.nn.functional.mse_loss(prediction[:, :, 8:-9, 38:-38] * masks_output, Y * masks_output)

        loss_size.backward()
        optim.step()

        # Print statistics
        running_loss += loss_size.item()
        if (i + 1) % config['print_every_step'] == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.3f} took: {:.2f}s".format(
                epoch + 1, int(100 * (i + 1) / n_batches), running_loss / config['print_every_step'],
                time.time() - start_time))

            # write the train loss to tensorboard
            running_loss_norm = running_loss / config['print_every_step']
            writer.write_loss_train(running_loss_norm, globaliter)
            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()

        if config['debug'] and i >= 0:
            break

    return globaliter


def validate(model, val_loader, device, config, globaliter=0, writer=None, mask=None, print_loss=True, norm=True):
    total_val_loss = 0
    # change to validation mode
    model.eval()

    if config['model']['depth'] <= 5:
        padd = torch.nn.ZeroPad2d((6, 6, 1, 0))
    elif config['model']['depth'] <= 7:
        padd = torch.nn.ZeroPad2d((6, 6, 8, 9))
    elif config['model']['depth'] == 8:
        padd = torch.nn.ZeroPad2d((38, 38, 8, 9))
    else:
        raise Exception("wrong padding for UNET depth")

    if mask is not None:
        mask_padd = padd(mask)

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):

            val_inputs, val_y, feature_dict = data
            if norm:
                val_inputs = val_inputs / 255

            # prepare input
            val_inputs = padd(val_inputs)
            val_inputs = val_inputs.float().to(device)

            # mask input
            if mask is not None:
                masks_input = mask_padd.expand(val_inputs.shape).float()
                val_inputs = val_inputs * masks_input

            # mask_output

            val_y = val_y.float().to(device)
            # masks_output = mask.expand(val_y.shape).float()

            val_output = model(val_inputs)

            # crop the output for comparing with true Y
            if mask is not None:
                masks_output = mask.expand(val_y.shape).float()
                if model.depth > 5:
                    val_loss_size = torch.nn.functional.mse_loss(val_output[:, :, 8:-9, 6:-6] * masks_output, val_y)
                else:
                    val_loss_size = torch.nn.functional.mse_loss(val_output[:, :, 1:, 6:-6] * masks_output, val_y)
            else:
                if model.depth > 5:
                    val_loss_size = torch.nn.functional.mse_loss(val_output[:, :, 8:-9, 6:-6], val_y)
                else:
                    val_loss_size = torch.nn.functional.mse_loss(val_output[:, :, 1:, 6:-6], val_y)

            total_val_loss += val_loss_size.item()

    val_loss = total_val_loss / len(val_loader)
    if print_loss:
        print("Validation loss = {:.2f}".format(val_loss))

    # write the validation loss to tensorboard
    if writer is not None:
        writer.write_loss_validation(val_loss, globaliter)

    return val_loss
