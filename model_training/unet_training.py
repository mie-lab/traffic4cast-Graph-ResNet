import os
import sys
from datetime import datetime
import torch
import logging
logging.basicConfig()

sys.path.append(os.getcwd())
from unet_config import config
from models.unet import UNet
from utils.training_unet_utils import trainNet
from utils.videoloader import trafic4cast_dataset
from utils.graph_utils import create_adj_matrix

if __name__ == "__main__":
    device = torch.device(config['device_num'])

    dataset_train = trafic4cast_dataset(split_type='training', **config['dataset'], reduce=True, filter_test_times=True)
    dataset_val = trafic4cast_dataset(split_type='validation', **config['dataset'], reduce=True, filter_test_times=True)

    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True,
                                               **config['dataloader'])
    val_loader = torch.utils.data.DataLoader(dataset_val, shuffle=True,
                                             **config['dataloader'])

    adj, nn_ixs, G, mask = create_adj_matrix(city=config['dataset']['cities'][0],
                                             mask_threshold=config['mask_threshold'])
    mask = torch.from_numpy(mask)
    mask = mask.to(device)

    for depth in [2, 3, 4, 5, 6]:
        config['model']['depth'] = depth

        print("===== HYPERPARAMETERS =====")
        print("batch_size=", config['dataloader']['batch_size'])
        print("epochs=", config['num_epochs'])
        print("learning_rate=", config['optimizer']['lr'])
        print("network_depth=", config['model']['depth'])
        print("=" * 30)

        log_folder = config['log_folder']
        log_dir = log_folder + 'Unet-depth_experiment' + '--' + str(config['model']['depth']) + '--' \
                  + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") + \
                  '-'.join(config['dataset']['cities'])

        model = UNet(**config['model']).to(device)
        trainNet(model, train_loader, val_loader, val_loader, device, log_dir=log_dir, config=config, mask=mask)
