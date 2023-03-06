from torchvision import transforms
import os
import torch
import torch.optim as optim


def init_loader(args):
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
    test_transform  = transforms.Compose([
        transforms.Resize(args.input_size),
        transforms.CenterCrop(args.crop_size),
        transforms.ToTensor(),
        transforms.Normalize(*stats)])

    # Set datasets
    if args.dataset == 'pacs':
        from Sagnet.data.pacs import PACS
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
    return loader_srcs, loader_vals, loader_tgts




def init_optimizer(args,model):
    global optimizer, optimizer_style, optimizer_adv
    global scheduler, scheduler_style, scheduler_adv
    global criterion, criterion_style, criterion_adv

    # Set hyperparams
    optim_hyperparams = {'lr': args.lr,
                         'weight_decay': args.weight_decay,
                         'momentum': args.momentum}
    if args.scheduler == 'step':
        Scheduler = optim.lr_scheduler.MultiStepLR
        sch_hyperparams = {'milestones': args.milestones,
                           'gamma': args.gamma}
    elif args.scheduler == 'cosine':
        Scheduler = optim.lr_scheduler.CosineAnnealingLR
        sch_hyperparams = {'T_max': args.iterations}

    # Main learning
    params = model.module.parameters()
    optimizer = optim.SGD(params, **optim_hyperparams)
    scheduler = Scheduler(optimizer, **sch_hyperparams)
    criterion = torch.nn.CrossEntropyLoss()

    if args.sagnet:
        # Style learning
        params_style = model.module.style_params()
        optimizer_style = optim.SGD(params_style, **optim_hyperparams)
        scheduler_style = Scheduler(optimizer_style, **sch_hyperparams)
        criterion_style = torch.nn.CrossEntropyLoss()

        # Adversarial learning
        params_adv = model.module.adv_params()
        optimizer_adv = optim.SGD(params_adv, **optim_hyperparams)
        scheduler_adv = Scheduler(optimizer_adv, **sch_hyperparams)
        criterion_adv = torch.nn.AdvLoss()

    return optimizer, optimizer_style, optimizer_adv, scheduler, scheduler_style, scheduler_adv, criterion, criterion_style, criterion_adv