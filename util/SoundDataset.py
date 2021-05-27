import pathlib
import numpy as np
import torch
from torch.utils.data import Dataset

class MultitaskDataset(Dataset):
    def __init__(self, datafn, labelfn, params, transform=None):
        self.transform = transform

        # make data and label name list
        path = pathlib.Path('.')
        datalist = list(path.glob(datafn+'*'))
        labellist = list(path.glob(labelfn+'*'))

        self.params = params
        self.datalist = datalist
        self.labellist = labellist
        self.datanum = len(datalist)

    def __len__(self):
        return self.datanum

    def __getitem__(self, idx):
        data = np.load(str(self.datalist[idx]))
        data = np.array(data['arr_0'],dtype='float32')
        data = torch.tensor(data,dtype=torch.float32)

        label = np.load(str(self.labellist[idx]))
        label = np.array(label['arr_0'],dtype='float32')
        label = torch.tensor(label,dtype=torch.float32)

        return data, label
