import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid

def get_dataset(args):
    """

    :param args:
    :return: train and test dataset and a user group
    """
    if args.dataset == 'mnist':
        data_dir = '../data/mnist'
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])

        train_dataset = datasets.MNIST(data_dir, train = True, download = True,
                                       transform=apply_transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download = True,
                                      transform=apply_transform)
        if args.iid:
            user_groups = mnist_iid(train_dataset, args.num_users)
        else :
            user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def init_loader():
    global loader_srcs, loader_vals, loader_tgts
    global num_classes

    # Set transforms
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    trans_list = []
    trans_list.append(transforms.RandomResizedCrop(args.crop_size, scale=(0.5, 1)))
    if args.colorjitter:
        trans_list.append(transforms.ColorJitter(*[args.colorjitter] * 4))
    trans_list.append(transforms.RandomHorizontalFlip())
    trans_list.append(transforms.ToTensor())
    trans_list.append(transforms.Normalize(*stats))

    train_transform = transforms.Compose(trans_list)
    test_transform = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats)])

    # Set datasets
    if args.dataset == 'pacs':
        from data.pacs import PACS
        image_dir = os.path.join(args.dataset_dir, args.dataset, 'images', 'kfold')
        split_dir = os.path.join(args.dataset_dir, args.dataset, 'splits')

        print('--- Training ---')
        dataset_srcs = [PACS(image_dir,
                             split_dir,
                             domain=domain,
                             split='train',
                             transform=train_transform)
                        for domain in args.sources]
        print('--- Validation ---')
        dataset_vals = [PACS(image_dir,
                             split_dir,
                             domain=domain,
                             split='crossval',
                             transform=test_transform)
                        for domain in args.sources]
        print('--- Test ---')
        dataset_tgts = [PACS(image_dir,
                             split_dir,
                             domain=domain,
                             split='test',
                             transform=test_transform)
                        for domain in args.targets]
        num_classes = 7
    else:
        raise NotImplementedError('Unknown dataset: {}'.format(args.dataset))

    # Set loaders
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    loader_srcs = [torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **kwargs)
        for dataset in dataset_srcs]
    loader_vals = [torch.utils.data.DataLoader(
        dataset,
        batch_size=int(args.batch_size * 4),
        shuffle=False,
        drop_last=False,
        **kwargs)
        for dataset in dataset_vals]
    loader_tgts = [torch.utils.data.DataLoader(
        dataset_tgt,
        batch_size=int(args.batch_size * 4),
        shuffle=False,
        drop_last=False,
        **kwargs)
        for dataset_tgt in dataset_tgts]



def average_weights(w):
    w_avg = copy.deepcopy(w[0]) #w[0] 깊은 복사
    for key in w_avg.keys():
        for i in range(1,len(w)):
            w_avg[key]+=w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w)) # 전체 개수로 나눠서 평균 취하기
    return w_avg

def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return