import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
import torchvision.transforms as transforms
import os
from util import *
import torch.cuda as cuda
import pickle
from torch.utils.data import Dataset, DataLoader



parser = argparse.ArgumentParser(description='Code for VISPE')
parser.add_argument('-e', '--epochs', action='store', default=300, type=int, help='epochs (default: 300)')
parser.add_argument('--batchSize', action='store', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--lr', '--learning-rate', action='store', default=0.001, type=float, help='learning rate (default: 0.001)')
parser.add_argument('--m', '--momentum', action='store', default=0.9, type=float, help='momentum (default: 0.9)')
parser.add_argument('--w', '--weight-decay', action='store', default=0, type=float, help='regularization weight decay (default: 0.0)')
parser.add_argument('--evaluate', action='store_true', default=False, help='Switch to evaluate mode (default: False)')
parser.add_argument('--gpu_num', type=int , default=0, help='gpu_num (default: 0)')
parser.add_argument('--load_pretrain', action='store_true', default=False, help='Flag to load pretrain model (default: False)')
parser.add_argument("--net", default='vgg16', const='vgg16',nargs='?', choices=['vgg16'], help="net model(default:vgg16)")
parser.add_argument("--dataset", default='modelnet40', const='modelnet40',nargs='?', choices=['modelnet40'], help="Dataset (default:modelnet40)")
parser.add_argument('--lamda', action='store', default=0.05, type=float, help='lamda (default: 0.05)')
parser.add_argument('--alpha', action='store', default=5, type=float, help='alpha (default: 5)')
parser.add_argument('--trial', action='store', default=1, type=int, help='trial (default: 1)')

arg = parser.parse_args()


def VISPE(model, x1, x2):
    criterion_KL = nn.KLDivLoss(reduction='batchmean')
    x1_feat = model(x1)
    x2_feat = model(x2)
    
    # prototype set 1
    x1_x2_mat = torch.exp(torch.matmul(x1_feat,x2_feat.t())/arg.lamda)
    denominator1 = torch.sum(x1_x2_mat, dim = 1).view(x1.shape[0],1)
    prob1 = x1_x2_mat/denominator1
    prob1_diag = torch.diag(prob1)

    # prototype set 2
    x2_x2_mat = torch.exp(torch.matmul(x2_feat,x2_feat.t())/arg.lamda)
    I = torch.eye(x1.shape[0]).cuda()
    x2_x2_mat = I*torch.diag(x1_x2_mat) + (1-I)*x2_x2_mat
    denominator2 = torch.sum(x2_x2_mat, dim = 1).view(x1.shape[0],1)
    prob2 = x2_x2_mat/denominator2    
    prob2_diag = torch.diag(prob2)
    
    # KL divergence
    loss_kl = criterion_KL(torch.log(prob1), prob2)    
    # cross entropy
    loss_ce = -torch.mean(torch.log(prob1_diag)+torch.log(prob2_diag))
    # Eq 8
    loss = loss_ce + arg.alpha *loss_kl  
    return loss

def remove_duplicate_object(x1, x2, obj_IDs):
    obj_set = set()
    obj_idx = []
    for i, oid in enumerate(obj_IDs):
        if oid not in obj_set:
            obj_idx.append(i)
            obj_set.add(oid)
    
    return x1[obj_idx], x2[obj_idx]

def main():
    # create model directory to store/load old model
    if not os.path.exists('model'):
        os.makedirs('model')
    if not os.path.exists('log'):
        os.makedirs('log')

	# Logger Setting
    logger = logging.getLogger('netlog')
    logger.setLevel(logging.INFO)
    if arg.load_pretrain:
        ch = logging.FileHandler('log/example.log')
    else:
        ch = logging.FileHandler('log/logfile_'+ arg.dataset + '_' + str(arg.trial)  + '.log')
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("================================================")
    logger.info("Learning Rate: {}".format(arg.lr))
    logger.info("Momentum: {}".format(arg.m))
    logger.info("Regularization Weight Decay: {}".format(arg.w))
    logger.info("Classifier: "+arg.net)
    logger.info("Dataset: "+arg.dataset)
    logger.info("Nbr of Epochs: {}".format(arg.epochs))
    logger.info("Lamda: {}".format(arg.lamda))
    logger.info("Alpha: {}".format(arg.alpha))
    logger.info("================================================")
    # Batch size setting
    batch_size = arg.batchSize
    
    # load the data
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    torch.cuda.set_device(arg.gpu_num)

    # dataset directory
    if arg.dataset == 'modelnet40':
        pickle_filename = './dataset/modelnet/seen.pickle'
        unseen_pickle_filename = './dataset/modelnet/unseen.pickle'
        k = 960 
        
    seen_train, seen_test = load_data(pickle_filename)
    unseen_train, unseen_test = load_data(unseen_pickle_filename)
    
    dataset = {}
    dataloader = {}
    if not arg.evaluate:
        dataset['train'] = mvDataset(seen_train, train=True, transform =data_transforms['train'])
        dataloader['train'] = DataLoader(dataset['train'], batch_size=arg.batchSize, shuffle=True, num_workers=4)
        
    dataset['seen_train_knn'] = mvDataset(seen_train, train=False, transform =data_transforms['test'])
    dataloader['seen_train_knn'] = DataLoader(dataset['seen_train_knn'], batch_size=arg.batchSize, shuffle=False, num_workers=4)        
    dataset['seen_test_knn'] = mvDataset(seen_test, train=False,  transform =data_transforms['test'])
    dataloader['seen_test_knn'] = DataLoader(dataset['seen_test_knn'], batch_size=arg.batchSize, shuffle=False, num_workers=4)
    
    dataset['unseen_train_knn'] = mvDataset(unseen_train, train=False, transform =data_transforms['test'])
    dataloader['unseen_train_knn'] = DataLoader(dataset['unseen_train_knn'], batch_size=arg.batchSize, shuffle=False, num_workers=4)        
    dataset['unseen_test_knn'] = mvDataset(unseen_test, train=False,  transform =data_transforms['test'])
    dataloader['unseen_test_knn'] = DataLoader(dataset['unseen_test_knn'], batch_size=arg.batchSize, shuffle=False, num_workers=4)
    
    if arg.net == 'vgg16':
        model = vgg16()

    optimizer = torch.optim.SGD(model.parameters(), lr=arg.lr, weight_decay=arg.w)
    model.cuda()
    
    model_path = 'model/model_'+ arg.dataset + '_' + str(arg.trial)  +'.pt'

    

    # training
    print("Start Training")
    logger.info("Start Training")
    epochs = arg.epochs if not arg.evaluate else 0
    min_acc = 0.0    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (x1, x2, obj_IDs) in enumerate(dataloader['train']):
            optimizer.zero_grad()
            x1, x2 = remove_duplicate_object(x1, x2, obj_IDs)
            
            loss = VISPE(model, x1.cuda(), x2.cuda())

            loss.backward()              
            optimizer.step()
            
            if batch_idx%50==0:
                print('==>>> epoch:{}, batch index: {}, loss:{}'.format(epoch, batch_idx, loss.cpu().detach().numpy()))
                logger.info('==>>> epoch:{}, batch index: {}, loss:{}'.format(epoch,batch_idx, loss.cpu().detach().numpy()))
            
        # Validation (always save the best model)
        print("Start Validation")
        logger.info("Start Validation")
        
        model.eval()
        seen_acc = kNN(0, model, dataloader['seen_train_knn'], dataloader['seen_test_knn'], k, len(dataset['seen_train_knn']), low_dim = 4096)
        unseen_acc = kNN(0, model, dataloader['unseen_train_knn'], dataloader['unseen_test_knn'], k, len(dataset['unseen_train_knn']), low_dim = 4096)
        if seen_acc >= min_acc:
            min_acc = seen_acc
            torch.save(model.state_dict(), model_path)
            
        print('==>>>test seen_acc:{} unseen_acc:{}'.format(seen_acc, unseen_acc))
        logger.info('==>>>test seen_acc:{} unseen_acc:{}'.format(seen_acc, unseen_acc))
        
            
    if arg.load_pretrain:
        if os.path.isfile('model/example.pt'):
            print("Loading pretrained model")
            model.load_state_dict(torch.load('model/example.pt', map_location=lambda storage, loc: storage))
        else:
            print("No model")
            return
    else:
        if os.path.isfile(model_path):
            print("Loading model")
            model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        else:
            print("No model")
            return
    
    model.eval()
    seen_acc = kNN(0, model, dataloader['seen_train_knn'], dataloader['seen_test_knn'], k, len(dataset['seen_train_knn']), low_dim = 4096)
    unseen_acc = kNN(0, model, dataloader['unseen_train_knn'], dataloader['unseen_test_knn'], k, len(dataset['unseen_train_knn']), low_dim = 4096)
    
    print('==>>>test seen_acc:{} unseen_acc:{}'.format(seen_acc, unseen_acc))
    logger.info('==>>>test seen_acc:{} unseen_acc:{}'.format(seen_acc, unseen_acc))
        
if __name__ == "__main__":
    main()
