import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset-dir', type=str, default='dataset',
                        help='home directory to dataset')
    parser.add_argument('--dataset', type=str, default='pacs',
                        help='dataset name')
    parser.add_argument('--model', type=str, default='sagnet',
                        help='model name')
    parser.add_argument('--sources', type=str, nargs=['art_painting', 'cartoon'],
                        help='domains for train')
    parser.add_argument('--targets', type=str, nargs='*',
                        help='domains for test')

    # save dir
    parser.add_argument('--save-dir', type=str, default='checkpoint',
                        help='home directory to save model')
    parser.add_argument('--method', type=str, default='sagnet',
                        help='method name')

    # data loader
    parser.add_argument('--workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size for each source domain')
    parser.add_argument('--input-size', type=int, default=256,
                        help='input image size')
    parser.add_argument('--crop-size', type=int, default=224,
                        help='crop image size')
    parser.add_argument('--colorjitter', type=float, default=0.4,
                        help='color jittering')

    # model
    parser.add_argument('--arch', type=str, default='sag_resnet',
                        help='network archiecture')
    parser.add_argument('--depth', type=str, default='18',
                        help='depth of network')
    parser.add_argument('--drop', type=float, default=0.5,
                        help='dropout ratio')

    # sagnet
    parser.add_argument('--sagnet', action='store_true', default=False,
                        help='use sagnet')
    parser.add_argument('--style-stage', type=int, default=3,
                        help='stage to extract style features {1, 2, 3, 4}')
    parser.add_argument('--w-adv', type=float, default=0.1,
                        help='weight for adversarial loss')

    # training policy
    parser.add_argument('--from-sketch', action='store_true', default=False,
                        help='training from scratch')
    parser.add_argument('--lr', type=float, default=0.004,
                        help='initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--iterations', type=int, default=2000,
                        help='number of training iterations')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        help='learning rate scheduler {step, cosine}')
    parser.add_argument('--milestones', type=int, nargs='+', default=[1000, 1500],
                        help='milestones to decay learning rate (for step scheduler)')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='gamma to decay learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--clip-adv', type=float, default=0.1,
                        help='grad clipping for adversarial loss')

    # etc
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='iterations for logging training status')
    parser.add_argument('--log-test-interval', type=int, default=10,
                        help='iterations for logging test status')
    parser.add_argument('--test-interval', type=int, default=100,
                        help='iterations for test')
    parser.add_argument('-g', '--gpu-id', type=str, default='0',
                        help='gpu id')

    # federated arguments
    parser.add_argument('--global-ep',type = int, default = 100,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=4,
                        help="number of users : K")
    parser.add_argument('--frac', type=float, default=0.5,
                        help="the fraction of clients : C")
    parser.add_argument('--local_ep', type=int, default=5,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=32,
                        help="local batch size : B")

    #others
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                               non-i.i.d setting (use 0 for equal splits)')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')

    args = parser.parse_args()
    return args

'''
    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                           use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                           of channels of imgs")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                           mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                           strided convolutions")
     '''


    # other arguments
    #parser.add_argument('--dataset', type=str, default='mnist', help="name \of dataset")
    #parser.add_argument('--num_classes', type=int, default=10, help="number \of classes")
    #parser.add_argument('--gpu', default=1, help="To use cuda, set \to a specific GPU ID. Default set to use CPU.")
    #parser.add_argument('--optimizer', type=str, default='sgd', help="type \of optimizer")

    #parser.add_argument('--verbose', type=int, default=1, help='verbose')
    #parser.add_argument('--seed', type=int, default=1, help='random seed')

