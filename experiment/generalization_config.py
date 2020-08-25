import os

config = dict()

# data set configuration
config['dataset'] = {}

config['dataset'][
    'source_root'] = r'./data/data_raw'
config['dataset'][
    'target_root'] = r'./data/data_proc'

config['dataset']['return_features'] = True
##################################################################


# data loader configuration
config['dataloader'] = {}
config['dataloader']['drop_last'] = False
config['dataloader']['num_workers'] = 4
config['dataloader']['batch_size'] = 1

# Graph creation
# I don't have any idea of an optimal value for the mask threshold. In the past I often used 50'000
# I noticed that 200'000 is probably too much. 
config['mask_threshold'] = 10

config['device_num'] = 0
config['debug'] = False

# A list of all models used in the generalization experiment. Models are defined via tuples:
# modeltuple[0]: Human readable name of the model for plots etc.
# modeltuple[1]: path of the folder where the model is stored.
# Models are stored during training as checkpoint.pt file via utils/earlystopping.py
# modeltuple[2]: Boolean flag indicating if it is a graph model
# modeltuple[3]: Name of the network in `models/graph_models.py` (only for graph models)

config['model_tuple_list'] = [
    ('KipfNet nh=16', os.path.join('.', 'runs', 'PMLR_nets', 'kipfnet16'), True, 'kipfnet'),
    ('KipfNet nh=128', os.path.join('.', 'runs', 'PMLR_nets', 'kipfnet128'), True, 'kipfnet'),
    ('Graph-ResNet', os.path.join('.', 'runs', 'PMLR_nets', 'graphresnet1'), True, 'Graph_resnet'),
    ('SkipfNet1', os.path.join('.', 'runs', 'PMLR_nets', 'skipfnet1_1'), True, 'skipfnet'),
    ('SkipfNet2', os.path.join('.', 'runs', 'PMLR_nets', 'skipfnet2_2'), True, 'skipfnet2d'),
    ('UNet depth=2', os.path.join('.', 'runs', 'PMLR_nets', 'Unet_Moscow_depth=2'), False, None),
    ('UNet depth=3', os.path.join('.', 'runs', 'PMLR_nets', 'Unet_Moscow_depth=3'), False, None),
    ('UNet depth=4', os.path.join('.', 'runs', 'PMLR_nets', 'Unet_Moscow_depth=4'), False, None,),
    ('UNet depth=5', os.path.join('.', 'runs', 'PMLR_nets', 'Unet_Moscow_depth=5'), False, None),
    ('UNet depth=6', os.path.join('.', 'runs', 'PMLR_nets', 'Unet_Moscow_depth=6'), False, None),
    ('UNet depth=5 MIE-Lab', os.path.join('.', 'runs', 'PMLR_nets', 'UNet_Moscow'), False, None),
]
