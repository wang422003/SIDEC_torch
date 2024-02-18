import time
import argparse
import pickle
import os
import datetime
import torch
import numpy as np


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--seed', type=int, default=1265, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Number of samples per batch.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Initial learning rate.')
    parser.add_argument('--hidden', type=int, default=4096,
                        help='Number of hidden units.')
    parser.add_argument('--suffix', type=str, default='',
                        help='Suffix for training data (e.g. "_charged".')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--save-folder', type=str, default='logs',
                        help='Where to save the trained model, leave empty to not save anything.')
    parser.add_argument('--load-folder', type=str, default='',
                        help='Where to load the trained model if finetunning. ' +
                             'Leave empty to train from scratch')
    parser.add_argument('--dims', type=int, default=4,
                        help='The number of input dimensions (position + velocity).')
    parser.add_argument('--timesteps', type=int, default=49,
                        help='The number of time steps per sample.')
    parser.add_argument('--prediction-steps', type=int, default=10, metavar='N',
                        help='Num steps to predict before re-using teacher forcing.')
    parser.add_argument('--lr-decay', type=int, default=200,
                        help='After how epochs to decay LR by a factor of gamma.')
    parser.add_argument('--gamma', type=float, default=0.5,
                        help='LR decay factor.')
    parser.add_argument('--var', type=float, default=5e-5,
                        help='Output variance.')
    parser.add_argument('--hard', action='store_true', default=False,
                        help='Uses discrete samples in training forward pass.')
    parser.add_argument('--prior', action='store_true', default=False,
                        help='Whether to use sparsity prior.')

    # for benchmark:
    parser.add_argument('--save-probs', action='store_true', default=False,
                        help='Save the probs during test.')
    parser.add_argument('--b-portion', type=float, default=1.0,
                        help='Portion of data to be used in benchmarking.')
    parser.add_argument('--b-time-steps', type=int, default=49,
                        help='Portion of time series in data to be used in benchmarking.')
    parser.add_argument('--b-shuffle', action='store_true', default=False,
                        help='Shuffle the data for benchmarking?.')
    parser.add_argument('--b-manual-nodes', type=int, default=0,
                        help='The number of nodes if changed from the original dataset.')
    parser.add_argument('--data-path', type=str, default='',
                        help='Where to load the data. May input the paths to edges_train of the data.')
    parser.add_argument('--b-network-type', type=str, default='',
                        help='What is the network type of the graph.')
    parser.add_argument('--b-directed', action='store_true', default=False,
                        help='Default choose trajectories from undirected graphs.')
    parser.add_argument('--b-simulation-type', type=str, default='',
                        help="Either springs or netsims or one from ['LI', 'LL', 'CY', 'BF', 'TF', 'BF-CV'].")

    parser.add_argument('--b-suffix', type=str, default='',
        help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
             ' Or "50r1" for 50 nodes, rep 1 and noise free.')
    # remember to disable this for submission
    parser.add_argument('--b-walltime', action='store_false', default=True,
                        help='Set wll time for benchmark training and testing. (Max time = 2 days)')

    # for yaml:
    parser.add_argument('--yaml', action='store_true', default=False,
                        help='If called, the args will be imported from /configs/vde.yaml.')
    # for VDE:
    parser.add_argument('--lag-time', type=int, default=1,
                        help='The lag time for VDE.')
    parser.add_argument('--slide', type=int, default=1,
                        help='The slide size for VDE.')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vde.yaml')

    # for test:
    parser.add_argument('--test', action='store_true', default=False,
                        help='If called, will load only 5% trajectories for training, validation and test.')

    # for ppcor:
    parser.add_argument('--use-future-latent', action='store_true', default=False,
                        help='If called, the latent features to be fed into ppcor will be remake with future prediction.')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    if args.use_future_latent:
        args.lag_time = 1
    else:
        args.lag_time = 0

    if args.test:
        args.b_portion = 0.05

    if args.suffix == "":
        args.suffix = args.b_simulation_type
        args.timesteps = args.b_time_steps

    if args.b_simulation_type in ['LI', 'LL', 'CY', 'BF', 'TF', 'BF-CV']:
        args.dims = 1
    elif args.b_simulation_type == 'springs':
        args.dims = 4
    elif args.b_simulation_type == 'netsims':
        args.dims = 1
    elif args.b_simulation_type == 'charged':
        args.dim = 4

    if args.data_path == "" and args.b_network_type != "":
        if args.b_simulation_type != 'charged' and args.b_simulation_type not in ['LI', 'LL', 'CY', 'BF', 'TF', 'BF-CV']:
            if args.b_directed:
                dir_str = 'directed'
            else:
                dir_str = 'undirected'
            args.data_path = os.path.dirname(os.getcwd()) + '/src/simulations/' + args.b_network_type + '/' + \
                             dir_str +\
                             '/' + args.b_simulation_type + '/edges_train_' + args.b_simulation_type + args.b_suffix + '.npy'
            args.b_manual_nodes = int(args.b_suffix.split('r')[0])
        elif args.b_simulation_type in ['LI', 'LL', 'CY', 'BF', 'TF', 'BF-CV']:
            args.b_directed = True
            args.data_path = os.path.dirname(os.getcwd()) + '/src/simulations/Synthetic-H/' + args.b_network_type + '/edges.npy'
            if args.b_simulation_type == 'LI':
                args.b_manual_nodes = 7
            elif args.b_simulation_type == 'LL':
                args.b_manual_nodes = 18
            elif args.b_simulation_type == 'CY':
                args.b_manual_nodes = 6
            elif args.b_simulation_type == 'BF':
                args.b_manual_nodes = 7
            elif args.b_simulation_type == 'TF':
                args.b_manual_nodes = 8
            elif args.b_simulation_type == 'BF-CV':
                args.b_manual_nodes = 10
        else:
            args.data_path = (os.path.dirname(os.getcwd()) + '/src/simulations/charged_particles/' +
                              '/edges_train_' + args.b_simulation_type + args.b_suffix + '.npy')
            args.b_manual_nodes = int(args.b_suffix.split('r')[0])

    if args.data_path != '':
        args.num_atoms = args.b_manual_nodes
    # if args.data_path != '':
    #     args.suffix = args.data_path.split('/')[-1].split('_', 2)[-1]

    print("suffix: ", args.suffix)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if args.cuda:
        args.device = 'gpu'
    else:
        args.device = 'cpu'

    # Save model and meta-data. Always saves in a new sub-folder.
    if args.save_folder:
        exp_counter = 0
        now = datetime.datetime.now()
        timestamp = now.isoformat()
        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)
        name_str = args.data_path.split('/')[-4] + '_' + args.data_path.split('/')[-3] + '_' + \
                   args.data_path.split('/')[-1].split('_', 2)[-1].split('.')[0]
        # save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
        save_folder = './{}/VDE-{}-exp{}/'.format(args.save_folder, name_str, timestamp)
        os.mkdir(save_folder)
        meta_file = os.path.join(save_folder, 'metadata.pkl')
        encoder_file = os.path.join(save_folder, 'encoder.pt')
        decoder_file = os.path.join(save_folder, 'decoder.pt')
        res_folder = save_folder + 'results/'
        os.mkdir(res_folder)
        log_file = os.path.join(save_folder, 'log.txt')
        log = open(log_file, 'w')

        pickle.dump({'args': args}, open(meta_file, "wb"))
        args.save_folder_path = save_folder
        args.res_folder_path = res_folder
        args.tb_name = save_folder.split('/')[2]
    else:
        print("WARNING: No save_folder provided!" +
              "Testing (within this script) will throw an error.")

    if args.prediction_steps > args.timesteps:
        args.prediction_steps = args.timesteps

    print(args)
    return args


if __name__ == "__main__":
    args_ = parse_args()
    print(args_)