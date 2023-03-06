import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
from Sagnet.modules.loss import *
from Sagnet.modules.utils import *
import sys
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import CNNMnist, Sagnet
from utils import get_dataset_mnist, average_weights, exp_details, init_loader_sag, init_optimizer

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_LAUNCH_BLOCKING"]='1'

if __name__ == '__main__':
    CUDA_LAUNCH_BLOCKING = 1
    start_time = time.time()

    '''
    #define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')
    '''
    args = args_parser()
    exp_details(args)

    #if args.gpu_id:
     #   torch.cuda.set_device(args.gpu_id)
    #else:
    device = 'cpu'

    # Set domains
    if args.dataset == 'pacs':
        all_domains = ['art_painting', 'cartoon', 'sketch', 'photo']

    if args.sources[0] == 'Rest':
        args.sources = [d for d in all_domains if d not in args.targets]
    if args.targets[0] == 'Rest':
        args.targets = [d for d in all_domains if d not in args.sources]

    if len(args.sources) == 2:  # 도메인 2개 2개 나누는 경우
        args.targets = [d for d in all_domains if d not in args.sources]

    # Set save dir
    save_dir = os.path.join(args.save_dir, args.dataset, args.method, ','.join(args.sources[0] + args.sources[1]))
    print('Save directory: {}'.format(save_dir))
    os.makedirs(save_dir, exist_ok=True)

    # Set Logger
    log_path = os.path.join(save_dir, 'log.txt')
    sys.stdout = Logger(log_path)

    # Print arguments
    print('\nArguments')
    for arg in vars(args):
        print(' - {}: {}'.format(arg, getattr(args, arg)))

    # Init seed
    if args.seed >= 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)


    # Initialize status
    src_keys = ['t_data', 't_net', 'l_c', 'l_s', 'l_adv', 'acc']
    status = OrderedDict([
        ('iteration', 0),
        ('lr', 0),
        ('src', OrderedDict([(k, AverageMeter()) for k in src_keys])),
        ('val_acc', OrderedDict([(domain, 0) for domain in args.sources])),
        ('mean_val_acc', 0),
        ('test_acc', OrderedDict([(domain, 0) for domain in args.targets])),
        ('mean_test_acc', 0),
    ])

    #load dataset and user groups
    #train_dataset, test_dataset, user_groups = get_dataset(args)

    #Bulid Model
    if args.model == 'cnn':
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
    elif args.model == 'sagnet':
        if args.dataset == 'pacs':
            global_model = Sagnet(args=args)

    else:
        exit('Error : unrecognized model')

    # Initialize optimizer & loaders
    print('\nInitialize optimizers...')
    Opti_dict = init_optimizer(args,global_model)

    print('\nInitialize loaders...')
    dataset_dict = init_loader_sag(args)

    # set the model to train and send it to device
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weight = global_model.state_dict()

    # training
    train_loss, train_accuracy = [],[]
    val_acc_list, net_list = [],[]
    cv_loss, cv_acc = [],[]
    print_every =2
    val_loss_pre, counter = 0,0

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)  # m <- max(C*K,1)
        # St <- (random set of m clients)
        idxs_users = np.random.choice(range(args.num_users), m, replace = False)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=dataset_dict,
                                      status=status, optimizer=Opti_dict)
            w = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch
            )
            local_weights.append(copy.deepcopy(w))
            #local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        #loss_avg = sum(local_losses) / len(local_losses)
        #train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=dataset_dict,status=status, optimizer=Opti_dict)
            acc = local_model.test(model=global_model)
            list_acc.append(acc)
            #list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch + 1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch + 1} global rounds:')
            #print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100 * train_accuracy[-1]))

    # Test inference after completion of training
    test_acc, test_loss = local_model.test(args, global_model)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'. \
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time() - start_time))

print(" ")
# Plot Loss curve
    # plt.figure()
    # plt.title('Training Loss vs Communication rounds')
    # plt.plot(range(len(train_loss)), train_loss, color='r')
    # plt.ylabel('Training loss')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))
    #
    # # Plot Average Accuracy vs Communication rounds
    # plt.figure()
    # plt.title('Average Accuracy vs Communication rounds')
    # plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    # plt.ylabel('Average Accuracy')
    # plt.xlabel('Communication Rounds')
    # plt.savefig('../save/fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.
    #             format(args.dataset, args.model, args.epochs, args.frac,
    #                    args.iid, args.local_ep, args.local_bs))

"""   
file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].npy'. \
    format(args.dataset, args.model, args.epochs, args.frac, args.iid,
           args.local_ep, args.local_bs)
np.save(file_name, train_accuracy)
"""