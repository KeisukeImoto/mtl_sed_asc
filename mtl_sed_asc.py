# -*- coding: utf-8 -*-
import random
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util import SoundDataset, modelio, timer
from nn_funcs import loss_funcs
from sed_util import evaluator

#eventname = ['(object) banging','(object) impact','(object) rustling','(object) snapping','(object) squeaking','bird singing','brakes squeaking','breathing','car','children','cupboard','cutlery','dishes','drawer','fan','glass jingling','keyboard typing','large vehicle','mouse clicking','mouse wheeling','people talking','people walking','washing dishes','water tap running','wind blowing']

# define network structure
class MultitaskNet(nn.Module):
    def __init__(self, params, device='cuda:0'):
        super(MultitaskNet, self).__init__()
        self.params = params
        self.device = device
        self.conv1 = nn.Conv2d(1,params['nfilter1'],kernel_size=(3,3),stride=params['stride'],padding=(1,1))
        self.conv2 = nn.Conv2d(params['nfilter1'],params['nfilter1'],kernel_size=(3,3),stride=params['stride'],padding=(1,1))
        self.conv3 = nn.Conv2d(params['nfilter1'],params['nfilter1'],kernel_size=(3,3),stride=params['stride'],padding=(1,1))
        self.bn1 = nn.BatchNorm2d(params['nfilter1'])
        self.bn2 = nn.BatchNorm2d(params['nfilter1'])
        self.bn3 = nn.BatchNorm2d(params['nfilter1'])
        self.bigru1 = nn.GRU(256,params['gruunit'],params['nGRUlayer'],bidirectional=True)
        self.fc1 = nn.Linear(params['gruunit']*2,params['fcunit1'])
        self.fc2 = nn.Linear(params['fcunit1'],params['nevent'])
        self.dropout1 = nn.Dropout(p=0.15)
        self.conv4 = nn.Conv2d(params['nfilter1'],params['nfilter2'],kernel_size=(3,3),stride=params['stride'],padding=(1,1))
        self.conv5 = nn.Conv2d(params['nfilter2'],params['nfilter2'],kernel_size=(3,3),stride=params['stride'],padding=(1,1))
        self.bn4 = nn.BatchNorm2d(params['nfilter2'])
        self.bn5 = nn.BatchNorm2d(params['nfilter2'])
        self.fc3 = nn.Linear(512,params['fcunit2'])
        self.fc4 = nn.Linear(params['fcunit2'],params['nscene'])
        self.dropout2 = nn.Dropout(p=0.15)

    def forward(self,x0):

        x1 = torch.reshape(x0,(-1,1,self.params['fdim'],self.params['slen']))
        batchsize = x1.size(0)

        ### shared layer ###
        # 1st shared CNN layer
        x1 = F.relu(self.bn1(self.conv1(x1)))
        x1 = F.max_pool2d(x1,(8,1))

        # 2nd shared CNN layer
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = F.max_pool2d(x2,(2,1))

        # 3rd shared CNN layer
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x3 = F.max_pool2d(x3,(2,1))

        ### event specific layer ###
        # bidirectional GRU layer
        x4 = x3.permute(3,0,1,2)
        x4 = x4.reshape(self.params['slen'],batchsize,-1)
        x4 = self.bigru1(x4)

        # dense and output layer
        x5 = x4[0].permute(1,0,2)
        x5 = F.relu(self.fc1(x5))
        x5 = self.dropout1(x5)
        x6 = self.fc2(x5)

        ### scene specific layer ###
        # 4th CNN layer
        x7 = F.relu(self.bn4(self.conv4(x3)))
        x7 = F.max_pool2d(x7,(1,25))

        # 5th CNN layer
        x8 = F.relu(self.bn5(self.conv5(x7)))
        x8 = F.max_pool2d(x8,(1,20))
        #x8 = F.max_pool2d(x8,kernel_size=x8.size()[2:])

        # dense and output layer
        x9 = x8.reshape(batchsize,-1)
        x9  = F.relu(self.fc3(x9))
        x9 = self.dropout2(x9)
        x10 = self.fc4(x9)

        return x6, x10

#def main(args):
@hydra.main(config_name='./conf/config')
def main(cfg:DictConfig) -> None:

    setting = cfg.setting
    params = cfg.params
    cwd = setting['cwd']

    # fix seed of rand functions
    if setting['FIXSEED']:
        random.seed(setting['seed'])
        np.random.seed(setting['seed'])
        torch.manual_seed(setting['seed'])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if setting['mode'] == 'train':

        # start timer
        timer.tic()

        # load dataset and set data loader
        traindata = SoundDataset.MultitaskDataset(setting['traindfn'],setting['trainlfn'],params)
        testdata = SoundDataset.MultitaskDataset(setting['testdfn'],setting['testlfn'],params)
        train_loader = torch.utils.data.DataLoader(traindata, batch_size=1, shuffle=True, drop_last=False)
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False, drop_last=False)

        # define network structure
        device = torch.device('cuda:' + str(setting['gpuid']) if torch.cuda.is_available() else 'cpu')
        model = MultitaskNet(params,device=device)
        model = model.to(device)
        print(model)

        # set loss function and optimizer
        criterion = loss_funcs.CEplusBCEWithLogitsLoss(alpha=params['alpha'],beta=params['beta'],reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(),lr=0.0002)

        # model training
        losslist = []; losslist_eval = []
        for ii in range(setting['nepoch']):
            lossval = 0.0
            model.train()
            for jj, data in enumerate(train_loader):

                feature,label = data[0].to(device),data[1].to(device)
                optimizer.zero_grad()
                output = model(feature)
                label = torch.reshape(label,(label.size(1),params['slen'],params['nevent']+1))
                loss = criterion(output,label)
                loss.backward()
                optimizer.step()
                lossval += loss.data

            # calculate loss for evaluate dataset
            loss_eval = 0.0
            model.eval()
            with torch.no_grad():
                for jj, data in enumerate(test_loader):
                    feature, label = data[0].to(device),data[1].to(device)
                    if feature.size(1)==params['nbatch']:
                        lossval_eval = 0.0
                        output = model(feature)
                        output1, output2 = output
                        label = torch.reshape(label,(params['nbatch'],params['slen'],params['nevent']+1))
                        loss_eval = criterion(output,label)
                        lossval_eval += loss_eval.data

            # print loss
            print('# epoch = %d: training loss: %.3f, test loss: %.3f' % (ii+1, lossval/len(train_loader), lossval_eval/len(test_loader)))
            losslist.append(lossval)
            losslist_eval.append(lossval_eval)

        # save model
        if setting['saveflag']:
            modelio.savemodel(model,cwd + 'result/'+setting['resname']+'/'+setting['resname']+'.model')

        # stop timer
        timer.toc()

    elif setting['mode'] == 'test' or setting['mode'] == 'eval' or setting['mode'] == 'evaluate':

        # load dataset and set data loader
        testdata = SoundDataset.MultitaskDataset(setting['testdfn'],setting['testlfn'],params)
        test_loader = torch.utils.data.DataLoader(testdata, batch_size=1, shuffle=False, drop_last=False)

        # define network structure
        device = torch.device('cuda:' + str(setting['gpuid']) if torch.cuda.is_available() else 'cpu')
        model = MultitaskNet(params,device=device)

        # load model
        modelio.loadmodel(model,cwd + 'result/'+setting['resname']+'/'+setting['resname']+'.model')
        model = model.to(device)
        print(model)


    # calculate sound event labels & their boundaries
    model.eval()
    with torch.no_grad():
        for ii, data in enumerate(test_loader):
            feature, label = data
            feature = feature.to(device)
            output = model(feature)
            output1, output2 = output
            output1 = torch.sigmoid(output1)

            # convert cuda tensor to cpu numpy
            event_output = output1.data.cpu().detach().numpy().copy()
            event_output = np.reshape(event_output,(-1,params['nevent']))
            scene_output = output2.data.cpu().detach().numpy().copy()

            label = torch.reshape(label,(label.size(1),params['slen'],params['nevent']+1))
            label = label.detach().numpy().copy()
            label = np.asarray(label.astype(int))
            event_label = label[:,:,1:]; scene_label = label[:,0,0]
            event_label = np.reshape(event_label,(-1,params['nevent']))

            # concatenate prediction results or labels
            if ii == 0:
                event_outputs = event_output
                event_labels = event_label
                scene_outputs = scene_output
                scene_labels = scene_label
            else:
                event_outputs = np.append(event_outputs,event_output,axis=0)
                event_labels = np.append(event_labels,event_label,axis=0)
                scene_outputs = np.append(scene_outputs,scene_output,axis=0)
                scene_labels = np.append(scene_labels,scene_label,axis=0)

    # predict acoustic scenes
    scene_predictions = np.argmax(scene_outputs,axis=1)
    evaluator.asc_evaluation(scene_predictions,scene_labels,printflg=True,saveflag=True,path=cwd + 'result/'+setting['resname']+'/'+setting['resname'])

    # predict sound events
    result = evaluator.SEDresult(event_outputs,event_labels,params,setting)
    result.sed_evaluation(printflag=True,saveflag=True,path=cwd + 'result/'+setting['resname']+'/'+setting['resname'])

if __name__ == '__main__':
    main()
