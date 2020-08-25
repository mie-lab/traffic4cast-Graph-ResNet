config = dict()

# data set configuration
config['dataset'] = {}

# data path
config['dataset']['source_root'] = './data/data_raw'
config['dataset']['target_root'] = './data/data_proc'

config['dataset']['cities'] = ['Moscow']

# logging
config['log_folder'] = './runs/graphnets/'

# number of tries for random search
config['nb_of_models'] = 100
config['device_num'] = 0
config['debug'] = False

# nb of channels (including the XY coordinate layers)
config['n_features'] = 38

# data loader configuration
config['dataloader'] = {}
config['dataloader']['drop_last'] = True
config['dataloader']['num_workers'] = 4
config['dataloader']['batch_size'] = 1

# Graph creation
config['mask_threshold'] = 10

# optimizer
config['optimizer_name'] = 'ADAM'
config['optimizer'] = {}
config['optimizer']['lr'] = 0.01
config['optimizer']['weight_decay'] = 0.0001

# # lr schedule
config['lr_step_size'] = 5
config['lr_gamma'] = 0.1

# early stopping
config['patience'] = 2
config['num_epochs'] = 10
config['print_every_step'] = 10
