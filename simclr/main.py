import os
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import utils
from model import Model
from utils import EarlyStopping
from tqdm import tqdm

def train_val(net, data_loader, train_optimizer=None):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()
    
    total_loss, total_num = 0.0, 0
    data_bar = tqdm(data_loader, leave=False)
    if is_train:
        
        for pos_1, pos_2, target in data_bar:
            pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
            feature_1, out_1 = net(pos_1)
            feature_2, out_2 = net(pos_2)
            # [2*B, D]
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)

            #mask
            sim_matirx = sim_matrix * (torch.ones(2*batch_size, device=sim_matrix.device)
                                       - torch.eye(2*batch_size,device=sim_matrix.device))

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()
            
            #total_num += batch_size
            #total_loss += loss.item() * batch_size
            #data_bar.set_description('Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
            data_bar.set_description('Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, loss.item()))
            
        
            


    else:
        with torch.no_grad():
            for pos_1, pos_2, target in data_bar:
                pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
                feature_1, out_1 = net(pos_1)
                feature_2, out_2 = net(pos_2)
                # [2*B, D]
                out = torch.cat([out_1, out_2], dim=0)
                # [2*B, 2*B]
                sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)

                #mask
                sim_matirx = sim_matrix * (torch.ones(2*batch_size, device=sim_matrix.device)
                                           - torch.eye(2*batch_size,device=sim_matrix.device))

                # compute loss
                pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
                # [2*B]
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()

                #total_num += batch_size
                #total_loss += loss.item() * batch_size
                data_bar.set_description('Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, loss.item()))

    return np.float(-torch.log(pos_sim).mean())



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR batch_size=128')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=512, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=500, type=int, help='Number of sweeps over the dataset to train')
   

    args = parser.parse_args()
    batch_size, epochs = args.batch_size, args.epochs
    temperature = args.temperature

    train_data = utils.CIFAR10Pair(root='./data', train=True, transform=utils.train_transform, download=True)
    train_loader, valid_loader = utils.create_datasets(batch_size, train_data)
    # model setup and optimizer config


    if not os.path.exists('results_batch{}_2'.format(batch_size)):
        os.mkdir('results_batch{}_2'.format(batch_size))
   
    results = {'train_sim': [], 'valid_sim':[]}
    
    model = Model().cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)
    least_loss = np.Inf
    
    for epoch in range(1, epochs + 1):
        train_loss = train_val(model, train_loader, optimizer)
        valid_loss = train_val(model, valid_loader)
        
        results['train_sim'].append(train_loss)
        results['valid_sim'].append(valid_loss)
        
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(f'results_batch{batch_size}_2/statistics.csv', index_label='epoch')
    '''    
        if valid_loss < least_loss:
            least_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), f'results_batch{batch_size}_2/model.pth')
       
    print("Best epoch: ", best_epoch)
    '''
                
