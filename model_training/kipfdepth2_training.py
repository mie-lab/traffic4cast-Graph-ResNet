import os
import sys
import warnings
from datetime import datetime
import torch
import logging
logging.basicConfig()

sys.path.append(os.getcwd())
from graphnets_config import config
from models.graph_models import KipfNetd2
from utils.graph_utils import create_adj_matrix, blockify_A, create_coordinate_channel, \
    create_edge_index_from_adjacency_matrix
from utils.training_gcn_utils import trainNet
from utils.videoloader import trafic4cast_dataset

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

if __name__ == "__main__":
    device = torch.device(config['device_num'])

    dataset_train = trafic4cast_dataset(split_type='training', **config['dataset'],
                                        reduce=True, filter_test_times=True)
    dataset_val = trafic4cast_dataset(split_type='validation', **config['dataset'],
                                      reduce=True, filter_test_times=True)

    train_loader = torch.utils.data.DataLoader(dataset_train, shuffle=True,
                                               **config['dataloader'])
    val_loader = torch.utils.data.DataLoader(dataset_val, shuffle=True,
                                             **config['dataloader'])


    batch_size = config['dataloader']['batch_size']
    n_features = config['n_features']

    coords = create_coordinate_channel(b=batch_size)

    adj, nn_ixs, G, mask = create_adj_matrix(city=config['dataset']['cities'][0],
                                             mask_threshold=config['mask_threshold'])

    if batch_size > 1:
        adj = blockify_A(adj, batch_size)

    edge_index = create_edge_index_from_adjacency_matrix(adj).to(device)
    nb_of_models = config['nb_of_models']

    for i in range(nb_of_models):
        # nh1 = random.choice([8, 16, 32, 48, 64])
        # nh2 = random.choice([8, 16, 32, 48])
        # K = random.choice([2, 4, 6, 8])
        # K_mix = random.choice([1, 2, 4])
        # inout_skipconn = random.choice([True, False])

        # set parameters as used in the paper:
        nh1 = 64
        nh2 = 32
        K = 6
        K_mix = 2
        inout_skipconn = True

        # Print all of the hyper parameters of the training iteration:
        print("===== HYPERPARAMETERS =====")
        print("batch_size=", batch_size)
        print("epochs =", config['num_epochs'])
        print("learning_rate =", config['optimizer']['lr'])
        print("mask_threshold =", config['mask_threshold'])
        print("nh1 =", nh1)
        print("nh2 =", nh2)
        print("K =", K)
        print("K_mix =", 'K_mix')
        print("inout_skipconn =", inout_skipconn)
        print("=" * 30)

        log_folder = config['log_folder']
        log_dir = log_folder + 'KipfNetd2' + '_nh1=' + str(nh1) \
                  + '_nh2=' + str(nh2) \
                  + '_K=' + str(K) + '_Kmix=' + str(K_mix) \
                  + '_skip_conn' + str(inout_skipconn) \
                  + '_' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-") \
                  + '-'.join(config['dataset']['cities'])

        model = KipfNetd2(num_features=n_features, num_classes=9,
                          nh1=nh1, nh2=nh2, K=K, K_mix=K_mix,
                          inout_skipconn=inout_skipconn).to(device)
        try:
            trainNet(model, train_loader, val_loader, device,
                     adj, nn_ixs, edge_index, coords=coords, config=config, log_dir=log_dir)

        except RuntimeError:
            print('Out of Memory error for ', nh1, K)

        # test
