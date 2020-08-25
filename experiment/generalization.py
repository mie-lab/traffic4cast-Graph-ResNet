import os
import sys
import torch
from collections import defaultdict
import json
import pickle
import logging
logging.basicConfig()

sys.path.append(os.getcwd())
from generalization_config import config
from utils.graph_utils import create_adj_matrix, blockify_A, create_coordinate_channel, \
    create_edge_index_from_adjacency_matrix
from utils.training_gcn_utils import validate
from utils.training_unet_utils import validate as validate_unet
from utils.videoloader import trafic4cast_dataset


from models.unet import UNet
from models.graph_models import KipfNet_orig, KipfNet, KipfNetd2, Graph_resnet



def get_graphdata_obj(inputs, edge_index, y, num_features=38, num_classes=9):
    graphdata = Data(x=inputs, edge_index=edge_index, y=y)

    return graphdata


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


if __name__ == "__main__":
    print('batch_size: ', config['dataloader']['batch_size'])
    print(config['device_num'])
    device = torch.device(config['device_num'])

    model_tuple_list = config['model_tuple_list']

    resultdict = defaultdict(dict)

    for city in ['Berlin', 'Moscow', 'Istanbul']:
        config['dataset']['cities'] = [city]

        dataset_val = trafic4cast_dataset(split_type='validation', **config['dataset'],
                                          reduce=True, filter_test_times=True)

        val_loader = torch.utils.data.DataLoader(dataset_val, shuffle=False,
                                                 **config['dataloader'])

        for model_tuple in model_tuple_list:
            model_plot_name = model_tuple[0]
            model_path = model_tuple[1]
            is_graph = model_tuple[2]
            graph_model_name = model_tuple[3]

            with open(os.path.join(model_path, 'config.json'), 'r') as f:
                model_config = json.load(f)

                adj, nn_ixs, G, mask = create_adj_matrix(city=config['dataset']['cities'][0],
                                                         mask_threshold=config['mask_threshold'])

                if not is_graph:
                    model_config['model']['batch_norm'] = True
                    model = UNet(**model_config['model']).to(device)
                    model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pt'),
                                                     map_location=device))

                    mask = torch.from_numpy(mask).to(device)

                    if 'MIE-Lab' in model_plot_name:
                        norm = False
                    else:
                        norm = True

                    val_loss = validate_unet(model=model, val_loader=val_loader, device=device, mask=mask,
                                             config=model_config, print_loss=False, norm=norm)

                if is_graph:

                    n_features = 38
                    batch_size = config['dataloader']['batch_size']
                    assert batch_size == 1, "batch_size should be 1 for graphs"

                    coords = create_coordinate_channel(b=batch_size)

                    if config['dataloader']['batch_size'] > 1:
                        adj = blockify_A(adj, config['dataloader']['batch_size'])

                    edge_index = create_edge_index_from_adjacency_matrix(adj)
                    edge_index = edge_index.to(device)

                    if graph_model_name == 'kipfnet':
                        model = KipfNet_orig(num_features=n_features,
                                             num_classes=9, **model_config['model']['KIPF']).to(device)
                        model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pt'),
                                                         map_location=device))

                    elif graph_model_name == 'skipfnet':
                        model = KipfNet(num_features=n_features,
                                        num_classes=9, **model_config['model']['KipfNet']).to(device)
                        model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pt'),
                                                         map_location=device))

                    elif graph_model_name == 'skipfnet2d':
                        model = KipfNetd2(num_features=n_features,
                                          num_classes=9, **model_config['model']['KipfNetd2']).to(device)
                        model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pt'),
                                                         map_location=device))

                    elif graph_model_name == 'Graph_resnet':
                        model = Graph_resnet(num_features=n_features,
                                             num_classes=9, **model_config['model']['Graph_resnet']).to(device)
                        model.load_state_dict(torch.load(os.path.join(model_path, 'checkpoint.pt'),
                                                         map_location=device))

                    mask = None
                    val_loss = validate(model=model, val_loader=val_loader, device=device,
                                        adj=adj, nn_ixs=nn_ixs, edge_index=edge_index, coords=coords,
                                        mask=mask, batch_size=batch_size, print_loss=False)

                print("Validation loss {}: {} = {:.2f}".format(city, model_plot_name, val_loss))
                resultdict[model_plot_name][city] = val_loss

                nb_params = get_n_params(model)
                resultdict[model_plot_name]['nb_params'] = nb_params

    pickle.dump(resultdict, open(os.path.join('.', 'output', 'data_generalization.p'), 'wb'))
