import os
import sys
import numpy as np
import torch
import json
import time
import warnings
from torch_geometric.data import Data

sys.path.append(os.getcwd())
from utils.graph_utils import image_to_vector, blockify_data, retransform_unblockify_target
from utils.earlystopping import EarlyStopping

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from utils.visual_TB import Visualizer


def trainNet(model, train_loader, val_loader, device, adj, nn_ixs, edge_index, config, log_dir, coords=None):
    """

    Args:
        model:
        train_loader:
        val_loader:
        device:
        adj:
        nn_ixs:
        edge_index:
        config:
        log_dir:
        coords:

    Returns:

    """

    # define the optimizer & learning rate
    optim = torch.optim.Adam(model.parameters(), **config['optimizer'])

    # scheduler = StepLR(optim, step_size=config['lr_step_size'], gamma=config['lr_gamma'])

    writer = Visualizer(log_dir)

    # dump config file
    with open(os.path.join(log_dir, 'config.json'), 'w') as fp:
        json.dump(config, fp)

    # Time for printing
    training_start_time = time.time()
    globaliter = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config['patience'], verbose=True)
    #    adj = adj.to(device)
    batch_size = config['dataloader']['batch_size']
    print_every_step = config['print_every_step']
    # Loop for n_epochs
    for epoch_idx, epoch in enumerate(range(config['num_epochs'])):

        writer.write_lr(optim, globaliter)

        # train for one epoch
        globaliter = train(model=model, train_loader=train_loader, optim=optim, device=device, writer=writer,
                           epoch=epoch, globaliter=globaliter, adj=adj, nn_ixs=nn_ixs, edge_index=edge_index,
                           batch_size=batch_size, coords=coords, print_every_step=print_every_step)

        # At the end of the epoch, do a pass on the validation set
        # val_loss = validate(model, val_loader, device, writer, globaliter, adj, nn_ixs, edge_index)
        val_loss = validate(model=model, val_loader=val_loader, device=device, adj=adj, nn_ixs=nn_ixs,
                            edge_index=edge_index, batch_size=batch_size, coords=coords,
                            writer=writer, globaliter=globaliter)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if config['debug'] and epoch_idx >= 0:
            break

        # scheduler.step()

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

    # remember to close writer
    writer.close()


def get_graphdata_obj(inputs, edge_index, y):
    graphdata = Data(x=inputs, edge_index=edge_index, y=y)

    return graphdata


def train(model, train_loader, optim, device, writer, \
          epoch, globaliter, adj, nn_ixs, edge_index, batch_size, print_every_step=1, coords=None):
    model.train()
    running_loss = 0.0
    n_batches = len(train_loader)

    # define start time
    start_time = time.time()

    for i, data in enumerate(train_loader, 0):

        inputs, Y, features = data
        inputs = inputs / 255
        globaliter += 1

        # the batch size of the last batch may vary, so we have to update the coords vector
        effective_batch_size = inputs.shape[0]
        if coords is not None and effective_batch_size != batch_size:
            coords_temp_list = [coords[0][0:effective_batch_size, ...], coords[1][0:effective_batch_size, ...]]

            coords = tuple(coords_temp_list)

        batch_size = effective_batch_size
        if coords is not None:
            inputs = torch.cat((inputs, coords[0], coords[1]), 1)

        inputs = image_to_vector(inputs, nn_ixs)
        Y = image_to_vector(Y, nn_ixs)

        inputs, Y = blockify_data(inputs, Y, batch_size)
        Y = Y.float()

        # the Y remains the same dimension
        inputs = inputs.float().to(device)
        Y = Y.float().to(device)

        graphdata = get_graphdata_obj(inputs, edge_index, Y)
        # Set the parameter gradients to zero
        optim.zero_grad()

        # Forward pass, backward pass, optimize
        prediction = model(graphdata)

        # crop the output for comparing with true Y
        loss_size = torch.nn.functional.mse_loss(prediction, Y)

        loss_size.backward()
        optim.step()

        # Print statistics
        running_loss += loss_size.item()
        if (i + 1) % print_every_step == 0:
            print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every_step,
                time.time() - start_time))

            # write the train loss to tensorboard
            running_loss_norm = running_loss / print_every_step
            writer.write_loss_train(running_loss_norm, globaliter)

            # Reset running loss and time
            running_loss = 0.0
            start_time = time.time()

    return globaliter


def validate(model, val_loader, device, adj, nn_ixs,
             edge_index, batch_size, writer=None, globaliter=None, coords=None, mask=None, print_loss=True):
    total_val_loss = 0
    loss_counter = 0
    debug_list = []

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):

            val_inputs_image, val_y1, feature_vec = data
            val_inputs_image = val_inputs_image / 255

            # the batch size of the last batch may vary, so we have to update the coords vector
            effective_batch_size = val_inputs_image.shape[0]
            if coords is not None and effective_batch_size != batch_size:
                coords_temp_list = [coords[0][0:effective_batch_size, ...], coords[1][0:effective_batch_size, ...]]

                coords = tuple(coords_temp_list)

            batch_size = effective_batch_size

            # add coords
            if coords is not None:
                val_inputs_image = torch.cat((val_inputs_image, coords[0], coords[1]), 1)

            val_inputs_vector = image_to_vector(val_inputs_image, nn_ixs)
            # tested
            val_y2 = image_to_vector(val_y1, nn_ixs)

            val_inputs_block, val_y_block = blockify_data(val_inputs_vector, val_y2, batch_size)

            # the Y remains the same dimension
            val_inputs_block = val_inputs_block.float().to(device)
            val_y_block = val_y_block.float().to(device)

            val_graphdata = get_graphdata_obj(val_inputs_block, edge_index, val_y_block)

            # Set the parameter gradients to zero
            # Forward pass, backward pass, optimize
            prediction_block = model(val_graphdata)

            # crop the output for comparing with true Y
            prediction_block = torch.clamp(prediction_block, 0, 255, out=None)

            # transform to image before calculating prediction error
            val_output = retransform_unblockify_target(prediction_block.cpu().detach().numpy(),
                                                       nn_ixs=nn_ixs,
                                                       batch_size=batch_size,
                                                       dataset=val_loader.dataset)
            val_output = torch.from_numpy(val_output).float().to(device)

            val_y1 = val_y1.float().to(device)

            assert not np.isnan(np.sum(val_output.cpu().detach().numpy()))
            assert not np.isnan(np.sum(val_y1.cpu().detach().numpy()))

            val_loss_size = torch.nn.functional.mse_loss(val_output, val_y1)

            total_val_loss += val_loss_size.item()
            loss_counter += 1
            debug_list.append(val_loss_size.item())

    val_loss = total_val_loss / loss_counter
    if print_loss:
        print("Validation loss = {:.2f}".format(val_loss))
    # write the validation loss to tensorboard
    if writer is not None:
        writer.write_loss_validation(val_loss, globaliter)
    return val_loss
