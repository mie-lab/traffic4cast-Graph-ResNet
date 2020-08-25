import ntpath
import os

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tabulate_events(inpath, outpath, tags):
    summary_iterators = [EventAccumulator(os.path.join(inpath, dname)).Reload() for dname in os.listdir(inpath)]

    #    tags = summary_iterators[0].Tags()['scalars']

    #
    #     for it in summary_iterators:
    #         print(it.Tags())
    #         assert it.Tags()['scalars'] == tags

    for tag in tags:
        for it in summary_iterators:
            try:
                run_name = ntpath.basename(it.path)
                csv_path = os.path.join(outpath, run_name + '_' + tag.replace('/', '_') + '.csv')
                pd.DataFrame(it.Scalars(tag)).to_csv(csv_path)
                print('Success: {} - {}'.format(tag, it.path))
            except KeyError:
                print('Error: {} - {}'.format(tag, it.path))
                pass


def transform_to_csv(inpath, outpath, tags=None):
    if tags is None:
        tags = ['Loss/train', 'Loss/validation_testtimes', 'Loss/validation']
    tabulate_events(inpath, outpath, tags)


if __name__ == '__main__':
    #    inpath = os.path.join(".", "graphs", "Graphresnet")
    #    outpath = os.path.join(".", "csv_files", "graphresnet_csv")
    #    outpath = os.path.join(".", "graphs", "kipf_hyperparameter_csv")

    # inpath = os.path.join(".", 'graphs', 'unet_depth_experiment')
    # outpath = os.path.join(".", 'csv_files', 'unet_depth_experiment_csv')
    tags = ['Loss/train', 'Loss/validation_testtimes', 'Loss/validation']
    inpath = r'C:\Users\henry\OneDrive\Programming\traffic4cast\runs\PMLR_nets'
    outpath = r'C:\Users\henry\OneDrive\Programming\traffic4cast\runs\PMLR_nets_csv'
    tabulate_events(inpath, outpath, tags)

#    inpath_list = glob.glob(os.path.join(".", "graphs", "*"))
#    for inpath in inpath_list:
#         tabulate_events(inpath, outpath, tags)
