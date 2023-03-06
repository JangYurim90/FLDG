import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import time
from Sagnet.modules.utils import *
from Sagnet.modules.loss import *
import copy


#from Sagnet.train import *


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int (i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self,item): # image와 label을 tensor 타입을 얻는다.
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)
        #return image.clone().detach(), label.clone().detach()
class LocalUpdate(object):
    def __init__(self, args,dataset,status,optimizer, idxs=None, logger=None):
        self.args = args
        self.logger = logger
        self.trainloader, self.validloader, self.testloader = self.train_val_test(
            dataset, list(idxs)
        )
        #data
        self.loader_vals = dataset['loader_vals']
        self.loader_tgts = dataset['loader_tgts']
        self.loader_srcs = dataset['loader_srcs']

        # data split과정
        self.device = 'cuda:0' #if args.gpu else 'cpu'
        # loss function은 NLL로 default
        self.criterion = nn.NLLLoss().to(self.device)
        self.status = status

        self.optimizer = optimizer['optimizer']
        self.optimizer_style = optimizer['optimizer_style']
        self.optimizer_adv = optimizer['optimizer_adv']

        self.criterion = optimizer['criterion']
        self.criterion_style = optimizer['criterion_style']
        self.criterion_adv = optimizer['criterion_adv']

        self.scheduler = optimizer['scheduler']
        self.scheduler_style = optimizer['scheduler_style']
        self.scheduler_adv = optimizer['scheduler_adv']

    def train_val_test(self, dataset, idxs):
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), # int(len(idxs_train)) self.args.local_bs
                                 batch_size=self.args.local_bs, shuffle=True)
        validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                 batch_size=int(len(idxs_val)/10), shuffle=False)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                batch_size=int(len(idxs_test)/10),shuffle = False)

        return trainloader, validloader, testloader

    def update_weights(self, model, global_round):
        '''
        # train model
        model.train()
        epoch_loss=[]

        # optimizer 설정
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),lr = self.args.lr, weight_decay = 1e-4)

        for iter in range(self.args.local_ep):
            batch_loss=[]
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)

                model.zero_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx*len(images),
                        len(self.trainloader.dataset),
                        100.* batch_idx / len(self.trainloader), loss.item())
                    )
                self.logger.add_scalar('loss',loss.item())
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)
        '''
        results = []
        for step in range(self.args.local_ep):
            self.train(model, step)

            if (step + 1) == self.args.local_ep:
                save_model(model, self.args.save_dir, 'latest')

                for i, domain in enumerate(self.args.sources):
                    print('Validation: {}'.format(domain))
                    self.status['val_acc'][domain] = self.test(self.loader_vals[i])
                for i, domain in enumerate(self.args.targets):
                    print('Test: {}'.format(domain))
                    self.status['test_acc'][domain] = self.test(self.loader_tgts[i])

                self.status['mean_val_acc'] = sum(self.status['val_acc'].values()) / len(self.status['val_acc'])
                self.status['mean_test_acc'] = sum(self.status['test_acc'].values()) / len(self.status['test_acc'])

                print('Val accuracy: {:.5f} ({})'.format(self.status['mean_val_acc'],
                                                         ', '.join(['{}: {:.5f}'.format(k, v) for k, v in
                                                                    self.status['val_acc'].items()])))
                print('Test accuracy: {:.5f} ({})'.format(self.status['mean_test_acc'],
                                                          ', '.join(['{}: {:.5f}'.format(k, v) for k, v in
                                                                     self.status['test_acc'].items()])))

                results.append(copy.deepcopy(self.status))
                save_result(results, self.args.save_dir)

        return model.state_dict()

    def train(self, model, step):
        with torch.autograd.detect_anomaly():
            global dataiter_srcs

            ## Initialize iteration
            model.train()
            self.optimizer.step()
            # scheduler.step()
            # if args.sagnet:
            #     scheduler_style.step()
            #     scheduler_adv.step()

            ## Load data
            tic = time.time()

            n_srcs = len(self.args.sources)
            if step == 0:
                dataiter_srcs = [None] * n_srcs
            data = [None] * n_srcs
            label = [None] * n_srcs
            for i in range(n_srcs):
                if step % len(self.loader_srcs[i]) == 0:
                    dataiter_srcs[i] = iter(self.loader_srcs[i])
                data[i], label[i] = next(dataiter_srcs[i])

            data = torch.cat(data)
            label = torch.cat(label)
            rand_idx = torch.randperm(len(data))
            data = data[rand_idx]
            label = label[rand_idx].cuda()

            time_data = time.time() - tic

            ## Process batch
            tic = time.time()

            # forward
            y, y_style = model(data)

            if self.args.sagnet:
                # learn style
                loss_style = self.criterion(y_style, label)
                self.optimizer_style.zero_grad()
                loss_style.backward(retain_graph=True)
                self.optimizer_style.step()

                # learn style_adv
                loss_adv = self.args.w_adv * self.criterion_adv(y_style)
                self.optimizer_adv.zero_grad()
                loss_adv.backward(retain_graph=True)
                if self.args.clip_adv is not None:
                    torch.nn.utils.clip_grad_norm_(model.module.adv_params(), self.args.clip_adv)
                self.optimizer_adv.step()

            # learn content
            loss = self.criterion(y, label)
            self.optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            self.scheduler.step()
            if self.args.sagnet:
                self.scheduler_style.step()
                self.scheduler_adv.step()
            time_net = time.time() - tic

            ## Update status
            self.status['iteration'] = step + 1
            self.status['lr'] = self.optimizer.param_groups[0]['lr']
            self.status['src']['t_data'].update(time_data)
            self.status['src']['t_net'].update(time_net)
            self.status['src']['l_c'].update(loss.item())
            if self.args.sagnet:
                self.status['src']['l_s'].update(loss_style.item())
                self.status['src']['l_adv'].update(loss_adv.item())
            self.status['src']['acc'].update(compute_accuracy(y, label))

            ## Log result
            if step % self.args.log_interval == 0:
                print('[{}/{} ({:.0f}%)] lr {:.5f}, {}'.format(
                    step, self.args.local_ep, 100. * step / self.args.local_ep, self.status['lr'],
                    ', '.join(['{} {}'.format(k, v) for k, v in self.status['src'].items()])))

    def test(self,model):
        model.eval()
        preds, labels = [], []
        for batch_idx, (data, label) in enumerate(self.loader_tgt):
            # forward
            with torch.no_grad():
                y, _ = model(data)

            # result
            preds += [y.data.cpu().numpy()]
            labels += [label.data.cpu().numpy()]

            # log
            if self.args.log_test_interval != -1 and batch_idx % self.args.log_test_interval == 0:
                print('[{}/{} ({:.0f}%)]'.format(
                    batch_idx, len(self.loader_tgt), 100. * batch_idx / len(self.loader_tgt)))

        # Aggregate result
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        acc = compute_accuracy(preds, labels)
        return acc

    def inference(self, model):
        """

        :param model:
        :return: inference accuracy and loss
        """

        model.eval()
        loss, total, correct = 0.0,0.0,0.0

        for batch_idx, (images, labels) in enumerate(self.testloader):
            images, labels = images.to(self.device), labels.to(self.device)

            # Inference
            outputs = model(images)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct / total
        return accuracy, loss

def test_inference(args, model, test_dataset):
    model.eval()
    loss, total, correct = 0.0,0.0,0.0

    device = 'cuda:0' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        #inference
        output = model(images)
        batch_loss = criterion(output, labels)
        loss += batch_loss.item()

        #prediction
        _, pred_labels = torch.max(output,1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct / total
    return accuracy, loss

