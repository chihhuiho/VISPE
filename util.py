import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models.vgg import model_urls as model_url_vgg
import torchvision.models as models
import pickle
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

# kNN function is modified from https://github.com/mangye16/Unsupervised_Embedding_Learning/blob/master/utils.py
def kNN(epoch, net, trainloader, testloader, K, ndata, low_dim = 4096):
    net.eval()
    total = 0
    correct_t = 0
    testsize = testloader.dataset.__len__()

    trainLabels = torch.LongTensor(trainloader.dataset.labels).cuda()
    trainFeatures = np.zeros((low_dim, ndata))
    
    trainFeatures = torch.Tensor(trainFeatures).cuda() 
    
    C = trainLabels.max() + 1
    C = np.int(C)
    
    with torch.no_grad():
        transform_bak = trainloader.dataset.transform
        trainloader.dataset.transform = testloader.dataset.transform
        temploader = torch.utils.data.DataLoader(trainloader.dataset, batch_size=32, shuffle=False, num_workers=4)
        for batch_idx, data in tqdm(enumerate(temploader)):
            (inputs, targets) = data                
            targets = targets.cuda()
            batchSize = inputs.size(0)
            features = net(inputs.cuda())
            trainFeatures[:, batch_idx*batchSize:batch_idx*batchSize+batchSize] = features.t()
           
    trainloader.dataset.transform = transform_bak
    top1 = 0.
    with torch.no_grad():
        retrieval_one_hot = torch.zeros(K, C).cuda()
        for batch_idx, (inputs, targets) in tqdm(enumerate(testloader)):            
            targets = targets.cuda()
            batchSize = inputs.size(0)  
            features = net(inputs.cuda())
            total += targets.size(0)
            dist = torch.mm(features, trainFeatures)
            yd, yi = dist.topk(K, dim=1, largest=True, sorted=True)
            candidates = trainLabels.view(1,-1).expand(batchSize, -1)
            retrieval = torch.gather(candidates, 1, yi)
            retrieval_one_hot.resize_(batchSize * K, C).zero_()
            retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
            probs = torch.sum(retrieval_one_hot.view(batchSize, -1 , C), 1)
            _, predictions = probs.sort(1, True)
            correct = predictions.eq(targets.data.view(-1,1))
            top1 = top1 + correct.narrow(1,0,1).sum().item()
            
    return top1*100./total 

class mvDataset:
    def __init__(self, data, train=True, transform = None, rand_rate=0.0):
        self.data = data
        self.img1 = data['img_pth']
        self.img2 = data['img2_pth']
        self.labels = data['labels']
        self.transform = transform
        self.train = train
        self.rand_rate = rand_rate

    def __len__(self):
        return len(self.img1)
    
    def __getitem__(self, idx):
        if self.train:
            # data path is in the form of dataset_root_path/objectID_viewID.jpg
            # viewID is in form of xxx.jpg
         
            img1 = Image.open(self.img1[idx]).convert('RGB')
            img2_pth = random.sample(self.img2[idx],1)[0]   
            img2 = Image.open(img2_pth).convert('RGB')
            obj_ID = img2_pth.split('/')[-1][:-4]
            
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            return (img1, img2, obj_ID)
        else:
            y = self.labels[idx]
            img1 = Image.open(self.img1[idx]).convert('RGB')
            if self.transform is not None:
                img1 = self.transform(img1)
            return (img1, y)
        
def load_data(pickle_filename):
    with open(pickle_filename, "rb") as input_file:
        data = pickle.load(input_file)        
    return data['train'], data['test']


class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()
        self.features = models.vgg16(pretrained=False).features             
        self.classifier1 = nn.Sequential(*list(models.vgg16(pretrained=False).classifier)[0:5])
        
    def forward(self, x):     
        x = self.features(x).view(x.shape[0], 25088)            
        x = self.classifier1(x).view(x.shape[0],4096)
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        return x

