config = dict()

##################################################################
# data set configuration
# source_root = raw data
# target_root = folder to store preprocessed data (preprocessing only done once)


config['dataset'] = {}

# data path
config['dataset']['source_root'] = './data/data_raw'
config['dataset']['target_root'] = './data/data_proc'

config['dataset']['cities'] = ['Moscow']

# logging
config['log_folder'] = './runs/unets/'

config['device_num'] = 0
config['debug'] = False

# model statistics
config['model'] = {}
config['model']['in_channels'] = 36  # 36 without and 38 with additional coordinates
config['model']['n_classes'] = 9
config['model']['wf'] = 6
config['model']['padding'] = True
config['model']['up_mode'] = 'upconv'  # up_mode (str): 'upconv' or 'upsample'.
config['model']['batch_norm'] = True
config['cont_model_path'] = None  # Use this to continue training a previously started model.

# data loader configuration
config['dataloader'] = {}
config['dataloader']['drop_last'] = True

config['dataloader']['num_workers'] = 4
config['dataloader']['batch_size'] = 4

config['mask_threshold'] = 10

# optimizer
config['optimizer'] = {}
config['optimizer']['lr'] = 0.02
config['optimizer']['momentum'] = 0.9
config['optimizer']['nesterov'] = True

# lr schedule
config['lr_step_size'] = 5
config['lr_gamma'] = 0.05

# early stopping
config['patience'] = 5

config['num_epochs'] = 10
config['print_every_step'] = 10
