# Hard Train......................
#!/usr/bin/env python
# coding: utf-8

# In[17]:


import time,os,sys
import math

import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random
from utils.utils2 import *
from models.GMM_Model5 import *
from utils.metric import *
from datetime import datetime, timedelta
import zipfile
import logging
logging.basicConfig(filename = "Inference.log", filemode='w', level = logging.DEBUG)
random.seed('a')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MCANN_I:

    def __init__(self, opt):

        self.logger = logging.getLogger()
        self.logger.info("I am logging...")
        self.opt = opt
        
        self.train_days = opt.input_len
        self.predict_days = opt.output_len  
        self.output_dim = opt.output_dim
        self.hidden_dim = opt.hidden_dim
        self.layer_dim = opt.layer        
        self.is_watersheds = opt.watershed
        self.is_prob_feature = opt.watershed
        self.TrainEnd = opt.model
        self.os = opt.oversampling   
        self.gmm_l = self.predict_days  #opt.gmm_len
        self.opt_hinter_dim = opt.watershed
        self.is_over_sampling = 1
        self.encoder = EncoderLSTM(self.opt).to(device)        
        self.decoder = DecoderLSTM(self.opt).to(device)
        self.expr_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
        self.val_dir = os.path.join(self.opt.outf, self.opt.name, 'val')
        self.test_dir = os.path.join(self.opt.outf, self.opt.name, 'test')

    def std_denorm_dataset(self, predict_y0, pre_y):
        
        a2 = r_log_std_denorm_dataset(self.mean, self.std, 0, predict_y0, pre_y)

        return a2

    def inference_test(self, x_test, y_input1):
                                
        y_predict = []
        d_out = torch.tensor([]).to(device)
        self.encoder.eval()
        self.decoder.eval()
        sig = nn.Sigmoid()

        with torch.no_grad():

            
            x_test = torch.from_numpy(np.array(x_test, np.float32)).to(device)
            y_input1 = torch.from_numpy(np.array(y_input1, np.float32)).to(device)

            h0 = torch.zeros(self.layer_dim, x_test.size(0),self.hidden_dim).to(device)
            c0 = torch.zeros(self.layer_dim, x_test.size(0),self.hidden_dim).to(device)
   
            encoder_h, encoder_c, ww = self.encoder(x_test)  
            out4 = self.decoder(y_input1, encoder_h, encoder_c, ww) 

            y_predict.extend(out4)
            y_predict = [y_predict[i].item() for i in range(len(y_predict))]
            y_predict = np.array(y_predict).reshape(1,-1) 
            
        return y_predict


    def model_load(self,zipf):       
        
        with zipfile.ZipFile(zipf, "r") as file:
            file.extract("Norm.txt")
        norm = np.loadtxt('Norm.txt',dtype=float,delimiter=None)
        os.remove('Norm.txt')
        print("norm is: ", norm)
        self.mean = norm[0]
        self.std = norm[1]
        self.R_mean = norm[2]
        self.R_std = norm[3]
         
        with zipfile.ZipFile(zipf, "r") as archive:
            with archive.open("MCANN_encoder.pt","r") as pt_file:
                self.encoder.load_state_dict(torch.load(pt_file), strict=False)    
        with zipfile.ZipFile(zipf, "r") as archive:
            with archive.open("MCANN_decoder.pt","r") as pt_file:
                self.decoder.load_state_dict(torch.load(pt_file), strict=False)                  
        with zipfile.ZipFile(zipf, "r") as archive:
            with archive.open("GMM.pt","r") as pt_file:
                self.gmm = torch.load(pt_file)  
        with zipfile.ZipFile(zipf, "r") as archive:
            with archive.open("GMM0.pt","r") as pt_file:
                self.gmm0 = torch.load(pt_file)  
        with zipfile.ZipFile(zipf, "r") as archive:
            with archive.open("GM3.pt","r") as pt_file:
                self.gm3 = torch.load(pt_file) 
                
    def get_data(self, test_point):

        # data prepare
        trainX = pd.read_csv('./data_provider/datasets/'+ self.opt.reservoir_sensor+'.tsv', sep='\t')
        trainX.columns = ["datetime", "value"] 
        trainX.sort_values('datetime', inplace=True),
        if self.opt.watershed == 0:
            R_X = trainX
        else:
            R_X = pd.read_csv('./data_provider/datasets/'+self.opt.rain_sensor+'.tsv', sep='\t')
            R_X.columns = ["id", "datetime", "value"] 
            R_X.sort_values('datetime', inplace=True)

        
        # read reservoir data        
        point = trainX[trainX["datetime"]==test_point].index.values[0]
        reservoir_data = trainX[point-self.train_days:point]["value"].values.tolist()
        pre_gt = np.array(trainX[point-1:point]["value"])
        pre_gt = pre_gt[0]
        gt = np.array(trainX[point:point+self.predict_days]["value"])
        if pre_gt is None:
            print("pre_gt is None, please fill it or switch to another time point.")
        NN = np.isnan(reservoir_data).any() 
        if NN:
            print("There is None value in the input sequence.")   
        
        # read rain data
        point = R_X[R_X["datetime"]==test_point].index.values[0]
        rain_data = R_X[point-self.train_days:point]["value"].values.tolist()
        NN = np.isnan(rain_data).any() 
        if NN:
            print("There is None value in the rain input sequence.")      
       
        return reservoir_data, rain_data, pre_gt, gt
    
    def test_single(self, test_point):
        
        reservoir_data, indicator_data, pre_gt, gt = self.get_data(test_point)  
        predict = self.predict(test_point, reservoir_data, indicator_data, pre_gt)
        
        return predict, gt
    
    def predict(self, test_point, reservoir_data, rain_data, pre_gt):
        
        time_str = test_point
        self.encoder.eval()
        self.decoder.eval()
        test_predict = np.zeros(self.predict_days*self.output_dim)
                                
        test_month = []
        test_day = []
        test_hour = []
        new_time = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        for i in range(self.predict_days):
            new_time_temp = new_time + timedelta(minutes=30)
            new_time = new_time.strftime("%Y-%m-%d %H:%M:%S")
            
            month = int(new_time[5:7])
            day = int(new_time[8:10])
            hour = int(new_time[11:13])
   
            test_month.append(month)
            test_day.append(day)
            test_hour.append(hour)  
            
            new_time = new_time_temp
            
        y2 = cos_date(test_month, test_day, test_hour) 
        y2 = [[ff] for ff in y2]
 
        y3 = sin_date(test_month, test_day, test_hour) 
        y3 = [[ff] for ff in y3]        
        
        y_input1 = np.array([np.concatenate((y2,y3),1)])
        
        # input dimension 1
        x_test = np.array(r_log_std_normalization_1(reservoir_data, self.mean, self.std), np.float32).reshape(self.train_days,-1)

        if  (self.opt_hinter_dim >= 1):
            x_test_rain = log_std_normalization_1(rain_data, self.R_mean, self.R_std)  
            x_test_rain = [[ff] for ff in  x_test_rain] 
            gmm_input = x_test_rain
        else:
            gmm_input = x_test        
        
        # input dimension 2
        weights3 = self.gm3.weights_
        data_prob3 = self.gm3.predict_proba(np.array(x_test)[:,0:1].reshape(-1,1))                        
        prob_in_distribution3 = data_prob3[:, 0] * weights3[0] + data_prob3[:, 1] * weights3[1] + data_prob3[:, 2] * weights3[2]
        prob_like_outlier3 = 1 - prob_in_distribution3
        prob_like_outlier3 = prob_like_outlier3.reshape(-1, 1)
        prob_like_outlier3 = np.array(prob_like_outlier3, np.float32).reshape(-1, 1)
        x_test = np.concatenate((x_test, prob_like_outlier3), 1)        
        
        # input dimension 3, 4, 5           
        self.gmm0_means = np.squeeze(self.gmm0.means_)
        weights3 = self.gmm0.weights_
        data_prob30 = self.gmm0.predict_proba(np.array(gmm_input)[:,0:1].reshape(-1,1))    
        order1 = np.argmax(weights3)
        d0 = data_prob30[:,order1].reshape(-1,1)
        order2 = np.argmin(weights3)
        d1 = data_prob30[:,order2].reshape(-1,1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        data_prob3 = np.concatenate((d0,d1),1)
        data_prob3 = np.concatenate((data_prob3,data_prob30[:,order3].reshape(-1,1)),1)
        recover_prob = np.array(data_prob3, np.float32)
        x_test = np.concatenate((x_test, recover_prob[:, 0:1]), 1)
        x_test = np.concatenate((x_test, recover_prob[:, 1:2]), 1)
        x_test = np.concatenate((x_test, recover_prob[:, 2:3]), 1)        
        
        # input dimension 6, 7, 8
        gmm_prob30 = self.gmm.predict_proba(np.array(x_test)[-1*self.gmm_l:,1:2].reshape(1,-1))
        order1 = np.argmin(self.gmm.weights_)
        d0 = gmm_prob30[:,order1].reshape(-1,1)
        order2 = np.argmax(self.gmm.weights_)
        d1 = gmm_prob30[:,order2].reshape(-1,1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        d2 = gmm_prob30[:,order3].reshape(-1,1)
        gmm_prob3 = np.concatenate((d0,d1),1)
        gmm_prob3 = np.concatenate((gmm_prob3,d2),1)  
        prob0 = gmm_prob3[:,0].reshape(-1,1).repeat(self.train_days,axis=1)
        prob0 = prob0.reshape(len(prob0),-1,1)
        prob1 = gmm_prob3[:,1].reshape(-1,1).repeat(self.train_days,axis=1)
        prob1 = prob1.reshape(len(prob1),-1,1)
        prob2 = gmm_prob3[:,2].reshape(-1,1).repeat(self.train_days,axis=1)
        prob2 = prob2.reshape(len(prob2),-1,1)
        prob = np.concatenate((prob0,prob1),2)
        prob = np.concatenate((prob,prob2),2)
        x_test = [x_test] 
        x_test = np.concatenate((x_test, prob), 2)      
               
        y_predict = self.inference_test(x_test, y_input1)
        y_predict = np.array(y_predict.tolist())[0]
        y_predict = [y_predict[i].item() for i in range(len(y_predict))]

        test_predict = np.array(self.std_denorm_dataset(y_predict, pre_gt))       

        diff_predict = []
        test_predict = (test_predict + abs(test_predict))/2 
        
        return test_predict

    def Inference(self):
        # Inference the whole test set
        test_set = pd.read_csv('./data_provider/datasets/test_timestamps_24avg.tsv',sep='\t')
        test_points = test_set["Hold Out Start"]
        count = 0
        pre = []
        gt = []
        for testP in test_points:
            predicted, ground_truth = self.test_single(testP)     
            pre.extend(predicted)
            gt.extend(ground_truth)
        metric_rolling('Every 3 days', pre, gt, rm=self.predict_days, inter=self.predict_days)
        metric_rolling('Every 8 hours', pre, gt, rm=8, inter=self.predict_days)