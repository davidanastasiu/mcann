import numpy as np
import random
import pandas as pd
from ..utils.utils2 import (
    diff_order_1,
    log_std_normalization,
    gen_month_tag,
    gen_time_feature,
    cos_date,
    sin_date,
    RnnDataset,
    r_log_std_normalization,
    r_log_std_normalization_1,
)
import os
import torch
from torch.utils.data import DataLoader
import sklearn  # unused?
from sklearn.mixture import GaussianMixture
from scipy import stats  # unused?
import zipfile  # unused?


class DS:

    def __init__(self, opt, trainX, R_X):
        #         super(DS, self).__init__()

        self.opt = opt
        self.trainX = trainX
        self.R_X = R_X
        self.mean = 0
        self.std = 0
        self.mini = 0
        self.R_mean = 0
        self.R_std = 0
        self.tag = []
        self.sensor_data = []
        self.diff_data = []
        self.data = []
        self.data_time = []
        self.sensor_data_norm = []
        self.sensor_data_norm1 = []

        self.R_sensor_data = []
        self.R_data = []
        self.R_data_time = []
        self.R_sensor_data_norm = []
        self.R_sensor_data_norm1 = []

        self.val_points = []
        self.test_points = []
        self.test_start_time = self.opt.test_start
        self.test_end_time = self.opt.test_end
        self.opt_hinter_dim = opt.watershed
        self.gm3 = GaussianMixture(
            n_components=3,
        )

        self.is_over_sampling = 0
        self.norm_percen = 0
        self.oversampling = int(opt.oversampling)
        self.iterval = opt.os_v

        self.train_days = self.opt.input_len
        self.predict_days = self.opt.output_len
        self.val_near_days = self.predict_days
        self.lens = self.train_days + self.predict_days + 1
        self.batch_size = opt.batchsize
        self.thre1 = 0
        self.thre2 = 0
        self.os_h = opt.os_s
        self.os_l = opt.os_s
        self.gmm_l = self.predict_days  # opt.gmm_len

        self.is_prob_feature = 1
        self.val_data_loader = []
        self.train_data_loader = []
        self.month = []
        self.day = []
        self.hour = []

        self.h_value = []
        self.sampled_h_value = []
        self.expr_dir = os.path.join(self.opt.outf, self.opt.name, "train")
        self.read_dataset()
        self.roll = 8

        # save mean and std of dataset
        norm = []
        norm.append(self.get_mean())
        norm.append(self.get_std())
        norm.append(self.get_R_mean())
        norm.append(self.get_R_std())
        np.savetxt(self.expr_dir + "/" + "Norm.txt", norm)
        norm = np.loadtxt(self.expr_dir + "/" + "Norm.txt", dtype=float, delimiter=None)
        print("norm is: ", norm)
        if self.opt.mode == "train":
            self.val_dataloader()
            self.train_dataloader()
        else:
            self.refresh_dataset(trainX, R_X)

    def get_trainX(self):

        return self.trainX

    def get_data(self):

        return self.data

    def get_diff_data(self):

        return self.diff_data

    def get_sensor_data(self):

        return self.sensor_data

    def get_sensor_data_norm(self):

        return self.sensor_data_norm

    def get_sensor_data_norm1(self):

        return self.sensor_data_norm1

    def get_R_data(self):

        return self.R_data

    def get_R_sensor_data(self):

        return self.R_sensor_data

    def get_R_sensor_data_norm(self):

        return self.R_sensor_data_norm

    def get_R_sensor_data_norm1(self):

        return self.R_sensor_data_norm1

    def get_val_data_loader(self):

        return self.val_data_loader

    def get_train_data_loader(self):

        return self.train_data_loader

    def get_val_points(self):

        return self.val_points

    def get_test_points(self):

        return self.test_points

    def get_mean(self):

        return self.mean

    def get_std(self):

        return self.std

    def get_R_mean(self):

        return self.R_mean

    def get_R_std(self):

        return self.R_std

    def get_month(self):

        return self.month

    def get_day(self):

        return self.day

    def get_hour(self):

        return self.hour

    def get_tag(self):

        return self.tag

    # Fetch dataset from data file, do preprocessing, generate a tag for the time series where 0 means None value, 1 means valid vauel
    def read_dataset(self):

        # read sensor data to vector
        start_num = self.trainX[
            self.trainX["datetime"] == self.opt.start_point
        ].index.values[0]
        print("for sensor ", self.opt.reservoir_sensor, "start_num is: ", start_num)
        idx_num = 0  # unused?
        # foot label of train_end
        train_end = (
            self.trainX[self.trainX["datetime"] == self.opt.train_point].index.values[0]
            - start_num
        )
        print("train set length is : ", train_end)

        # the whole dataset
        self.sensor_data = self.trainX[
            start_num: train_end + start_num
        ]  # e.g. 2011/7/1  22:30:00 - 2020/6/22  23:30:00
        self.data = np.array(self.sensor_data["value"].fillna(np.nan))
        self.diff_data = diff_order_1(self.data)
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan))
        self.sensor_data_norm, self.mean, self.std, self.mini = r_log_std_normalization(
            self.data
        )
        self.sensor_data_norm1 = [[ff] for ff in self.sensor_data_norm]

        if self.opt_hinter_dim >= 1:
            # read Rain data to vector
            R_start_num = self.R_X[
                self.R_X["datetime"] == self.opt.start_point
            ].index.values[0]
            print("for sensor ", self.opt.rain_sensor, "start_num is: ", R_start_num)
            R_idx_num = 0  # unused?
            R_train_end = (
                self.R_X[self.R_X["datetime"] == self.opt.train_point].index.values[0]
                - R_start_num
            )
            print("R_X set length is : ", R_train_end)
            self.R_sensor_data = self.R_X[
                R_start_num: R_train_end + R_start_num
            ]  # e.g. 2011/7/1  22:30:00 - 2020/6/22  23:30:00
            self.R_data = np.array(self.R_sensor_data["value"].fillna(np.nan))
            self.R_data_time = np.array(self.R_sensor_data["datetime"].fillna(np.nan))
            self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(
                self.R_data
            )
            self.R_sensor_data_norm1 = [[ff] for ff in self.R_sensor_data_norm]
            gmm_input = self.R_sensor_data_norm
        else:
            gmm_input = self.sensor_data_norm

        if self.is_prob_feature == 1:
            clean_data = []
            for ii in range(len(self.sensor_data_norm)):
                if (self.sensor_data_norm[ii] is not None) and (
                    np.isnan(self.sensor_data_norm[ii]) != 1
                ):
                    clean_data.append(self.sensor_data_norm[ii])
            sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)
            # dataset-wise gmm
            self.gm3.fit(sensor_data_prob)
            torch.save(self.gm3, self.expr_dir + "/" + "GM3.pt")
            self.gm_means = np.squeeze(self.gm3.means_)
            self.z0 = np.min(self.gm_means)
            self.z1 = np.median(self.gm_means)
            self.z2 = np.max(self.gm_means)

            self.thre1 = (self.z0 + self.z1) / 2
            self.thre2 = (self.z1 + self.z2) / 2
            print("gm3.means are: ", self.gm_means)
            print("z : ", self.z0, self.z1, self.z2)
            print("gm3.covariances are: ", self.gm3.covariances_)
            print("gm3.weights are: ", self.gm3.weights_)
            weights3 = self.gm3.weights_
            data_prob3 = self.gm3.predict_proba(sensor_data_prob)

            prob_in_distribution3 = (
                data_prob3[:, 0] * weights3[0]
                + data_prob3[:, 1] * weights3[1]
                + data_prob3[:, 2] * weights3[2]
            )

            prob_like_outlier3 = 1 - prob_in_distribution3
            prob_like_outlier3 = prob_like_outlier3.reshape((len(sensor_data_prob), 1))
            print("data_prob3 shape, ", np.array(data_prob3).shape)
            recover_data = []
            temp = 0
            jj = 0
            for ii in range(len(self.sensor_data_norm)):
                if (self.sensor_data_norm[ii] is not None) and (
                    np.isnan(self.sensor_data_norm[ii]) != 1
                ):
                    recover_data.append(prob_like_outlier3[jj])
                    jj = jj + 1
                else:
                    recover_data.append(self.sensor_data_norm[ii])
            prob_like_outlier3 = np.array(recover_data, np.float32).reshape(
                len(self.sensor_data_norm), 1
            )
            self.sensor_data_norm1 = np.concatenate(
                (self.sensor_data_norm1, prob_like_outlier3), 1
            )

            # point-wise probability feture, generate dim 2-4
            clean_data = []
            for ii in range(len(gmm_input)):
                if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                    clean_data.append(gmm_input[ii])
            sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)
            # dataset-wise gmm
            self.gmm0 = GaussianMixture(
                n_components=3,
            )
            series = []
            random.seed(self.opt.val_seed)
            for ggg in range(200000):
                g0 = random.randint(0, len(gmm_input) - self.gmm_l)
                if not np.isnan(gmm_input[g0]).any():
                    series.append([gmm_input[g0]])
            self.gmm0.fit(np.array(series).reshape(-1, 1))
            torch.save(self.gmm0, self.expr_dir + "/" + "GMM0.pt")
            self.gmm0_means = np.squeeze(self.gmm0.means_)
            print("gmm0.means are: ", self.gmm0_means)
            print("gmm0.weights are: ", self.gmm0.weights_)
            weights3 = self.gmm0.weights_
            data_prob30 = self.gmm0.predict_proba(sensor_data_prob)
            order1 = np.argmax(weights3)
            d0 = data_prob30[:, order1].reshape(-1, 1)
            order2 = np.argmin(weights3)
            d1 = data_prob30[:, order2].reshape(-1, 1)
            for oi in range(3):
                if oi != order1 and oi != order2:
                    order3 = oi
            print("new order is, ", order1, order2, order3)
            data_prob3 = np.concatenate((d0, d1), 1)
            data_prob3 = np.concatenate(
                (data_prob3, data_prob30[:, order3].reshape(-1, 1)), 1
            )

            prob_in_distribution3 = (
                data_prob30[:, 0] * weights3[0]
                + data_prob30[:, 1] * weights3[1]
                + data_prob30[:, 2] * weights3[2]
            )

            prob_like_outlier3 = 1 - prob_in_distribution3
            prob_like_outlier3 = prob_like_outlier3.reshape((len(sensor_data_prob), 1))
            print("data_prob3 shape, ", np.array(data_prob3).shape)
            recover_data = []
            recover_prob = []
            temp = np.zeros(np.array(data_prob3[0]).shape)
            jj = 0
            for ii in range(len(gmm_input)):
                if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                    recover_data.append(prob_like_outlier3[jj])
                    recover_prob.append(data_prob3[jj])
                    jj = jj + 1
                else:
                    recover_data.append(gmm_input[ii])
                    recover_prob.append(temp)
            prob_like_outlier3 = np.array(recover_data, np.float32).reshape(
                len(gmm_input), 1
            )
            recover_prob = np.array(recover_prob, np.float32)
            self.sensor_data_norm1 = np.concatenate(
                (self.sensor_data_norm1, recover_prob[:, 0:1]), 1
            )
            self.sensor_data_norm1 = np.concatenate(
                (self.sensor_data_norm1, recover_prob[:, 1:2]), 1
            )
            self.sensor_data_norm1 = np.concatenate(
                (self.sensor_data_norm1, recover_prob[:, 2:3]), 1
            )
            print("sensor_data_norm1, ", self.sensor_data_norm1)
            print("Finish prob indicator generating.")

        if self.opt_hinter_dim < 1:
            self.R_data = prob_like_outlier3.squeeze()  # diff_order_1(self.data)
            print("R_data, ", self.R_data)
            #             self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(self.R_data)
            self.R_sensor_data_norm = self.R_data
            self.R_mean = 0
            self.R_std = 1
            self.R_sensor_data_norm1 = prob_like_outlier3.squeeze()
            self.R_sensor_data_norm = self.R_sensor_data_norm1

        self.tag = gen_month_tag(self.sensor_data)
        print("self.tag len, ", len(self.tag))

        self.month, self.day, self.hour = gen_time_feature(self.sensor_data)

        cos_d = cos_date(self.month, self.day, self.hour)
        cos_d = [[x] for x in cos_d]
        sin_d = sin_date(self.month, self.day, self.hour)
        sin_d = [[x] for x in sin_d]

    # Randomly choose a point in timesequence,
    # if it is a valid start time (with no nan value in the whole sequence, between Sep and May), tag it as 3
    # For those points near this point (near is defined as a parameter), tag them as 4

    def val_dataloader(self):

        print("Begin to generate val_dataloader!")
        DATA = []  # unused?
        Label = []  # unused?

        #         near_len = max(self.train_days, self.predict_days, self.val_near_days**4) # Avoid the left near_len and the right near_len points to train

        near_len = self.predict_days

        random.seed(self.opt.val_seed)

        if self.is_over_sampling == 1:

            print(
                "Over sampling on validation set is not supported by now, please wait..."
            )

        else:
            ii = 0
            while ii < self.opt.val_size:

                i = random.randint(self.predict_days, len(self.data) - self.lens - 1)
                a1 = 0
                a2 = -13
                if (
                    (not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any())
                    and (not np.isnan(self.R_data[i: i + self.lens]).any())
                    and (
                        self.tag[i + self.train_days] <= a1
                        or a2 < self.tag[i + self.train_days] < 0
                        or 2 <= self.tag[i + self.train_days] <= 3
                    )
                ):

                    self.tag[i + self.train_days] = 2  # tag 2 means in validation set

                    for k in range(near_len):
                        self.tag[i + self.train_days - k] = (
                            3  # tag 3 means near points of validation set
                        )
                        self.tag[i + self.train_days + k] = 3

                    point = self.data_time[i + self.train_days]
                    self.val_points.append([point])
                    ii = ii + 1

        self.opt.name = "%s" % (self.opt.model)
        val_dir = os.path.join(self.opt.outf, self.opt.name, "val")
        file_name = os.path.join(val_dir, "validation_timestamps_24avg.tsv")

        pd_temp = pd.DataFrame(data=self.val_points, columns=["Hold Out Start"])
        pd_temp.to_csv(file_name, sep="\t")
        print("val set saved to : ", file_name)

    # Can only be run after val_dataloader
    # Randomly choose a point in timesequence,
    # if it is a valid start time (with no nan value in the whole sequence between Sep and May, and tag is not 3 and 4),
    # select it as a train point, tag it as 5

    def train_dataloader(self):

        print("Begin to generate train_dataloader!")
        DATA = []
        Label = []

        # randomly choose train data
        random.seed(self.opt.train_seed)

        if self.is_over_sampling == 1:

            print(
                "Over sampling on validation set is not supported by now, please wait..."
            )

        else:
            ii = 0
            jj = 0
            while ii < self.opt.train_volume:

                i = random.randint(
                    self.predict_days * 4,
                    len(self.sensor_data_norm) - 31 * self.predict_days * 4 - 1,
                )
                pre1 = np.array(
                    self.sensor_data_norm[
                        (i + self.train_days): (
                            i + self.train_days + self.predict_days
                        )
                    ]
                )
                a1 = 0
                a2 = -13
                if np.max(pre1) > self.thre2:
                    a3 = self.os_h
                    max_index = np.argmax(pre1)
                elif np.min(pre1) < self.thre1:
                    a3 = self.os_l
                    max_index = np.argmin(pre1)
                a5 = self.iterval
                if (
                    (jj < self.opt.train_volume * (self.oversampling / 100))
                    and (np.max(pre1) > self.thre2 or np.min(pre1) < self.thre1)
                    and (not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any())
                    and (
                        self.tag[i + self.train_days] <= a1
                        or a2 < self.tag[i + self.train_days] < 0
                    )
                ):
                    if a3 > 0:
                        i = i + max_index - 1
                        i = i - a3 * a5
                    for kk in range(a3):  # (int(self.predict_days/self.iterval)):
                        i = i + a5
                        if (
                            i > len(self.data) - 31 * self.predict_days * 4 - 1
                            or i < self.predict_days * 4
                        ):
                            continue
                        if (
                            not np.isnan(
                                self.sensor_data_norm1[i: i + self.lens]
                            ).any()
                            and self.tag[i + self.train_days] != 2
                            and self.tag[i + self.train_days] != 3
                            and self.tag[i + self.train_days] != 4
                        ):

                            data0 = np.array(
                                self.sensor_data_norm1[i: (i + self.train_days)]
                            ).reshape(self.train_days, -1)
                            label00 = np.array(
                                self.sensor_data_norm[
                                    (i + self.train_days): (
                                        i + self.train_days + self.predict_days
                                    )
                                ]
                            )
                            label01 = np.array(
                                self.diff_data[
                                    (i + self.train_days): (
                                        i + self.train_days + self.predict_days
                                    )
                                ]
                            )
                            label01 = label00  # .astype(np.int)
                            label0 = [[ff] for ff in label01]

                            b = i + self.train_days
                            e = i + self.train_days + self.predict_days

                            label2 = cos_date(
                                self.month[b:e], self.day[b:e], self.hour[b:e]
                            )  # represent cos(int(data)) here
                            label2 = [[ff] for ff in label2]

                            label3 = sin_date(
                                self.month[b:e], self.day[b:e], self.hour[b:e]
                            )  # represent sin(int(data)) here
                            label3 = [[ff] for ff in label3]

                            label4 = np.array(
                                self.data[
                                    (i + self.train_days - 1): (
                                        i + self.train_days + self.predict_days - 1
                                    )
                                ]
                            ).reshape(-1, 1)
                            label5 = np.array(
                                self.data[
                                    (i + self.train_days): (
                                        i + self.train_days + self.predict_days
                                    )
                                ]
                            ).reshape(-1, 1)

                            label = np.concatenate((label0, label2), 1)
                            label = np.concatenate((label, label3), 1)
                            label = np.concatenate((label, label4), 1)
                            label = np.concatenate((label, label5), 1)

                            self.tag[i + self.train_days] = 4
                            jj = jj + 1
                            DATA.append(data0)
                            Label.append(label)
                #                     ii = ii + 1

                if (not np.isnan(self.sensor_data_norm1[i: i + self.lens]).any()) and (
                    self.tag[i + self.train_days] <= a1
                    or a2 < self.tag[i + self.train_days] < 0
                ):

                    if 1 == 1:
                        data0 = np.array(
                            self.sensor_data_norm1[i: (i + self.train_days)]
                        ).reshape(self.train_days, -1)
                        label00 = np.array(
                            self.sensor_data_norm[
                                (i + self.train_days): (
                                    i + self.train_days + self.predict_days
                                )
                            ]
                        )
                        label01 = np.array(
                            self.diff_data[
                                (i + self.train_days): (
                                    i + self.train_days + self.predict_days
                                )
                            ]
                        )
                        label01 = label00  # .astype(np.int)
                        label0 = [[ff] for ff in label01]

                        b = i + self.train_days
                        e = i + self.train_days + self.predict_days

                        label2 = cos_date(
                            self.month[b:e], self.day[b:e], self.hour[b:e]
                        )  # represent cos(int(data)) here
                        label2 = [[ff] for ff in label2]

                        label3 = sin_date(
                            self.month[b:e], self.day[b:e], self.hour[b:e]
                        )  # represent sin(int(data)) here
                        label3 = [[ff] for ff in label3]

                        label4 = np.array(
                            self.data[
                                (i + self.train_days - 1): (
                                    i + self.train_days + self.predict_days - 1
                                )
                            ]
                        ).reshape(-1, 1)
                        label5 = np.array(
                            self.data[
                                (i + self.train_days): (
                                    i + self.train_days + self.predict_days
                                )
                            ]
                        ).reshape(-1, 1)

                        label = np.concatenate((label0, label2), 1)
                        label = np.concatenate((label, label3), 1)
                        label = np.concatenate((label, label4), 1)
                        label = np.concatenate((label, label5), 1)

                        DATA.append(data0)
                        Label.append(label)

                        self.tag[i + self.train_days] = 4
                        ii = ii + 1

        self.DATA = DATA

        # sample-wise gmm, generate dim 5-7
        self.gmm = GaussianMixture(
            n_components=3,
        )
        xx = np.array(self.DATA, np.float32)
        self.gmm.fit(np.squeeze(xx[:, -1 * self.gmm_l:, 1:2]))
        torch.save(self.gmm, self.expr_dir + "/" + "GMM.pt")
        self.gmm_means = np.squeeze(self.gmm.means_)
        print("time series gmm.weights are: ", self.gmm.weights_)
        gmm_prob30 = self.gmm.predict_proba(
            np.squeeze(np.array(self.DATA)[:, -1 * self.gmm_l:, 1:2])
        )
        print("gmm_prob30, ", gmm_prob30)

        order1 = np.argmin(self.gmm.weights_)
        d0 = gmm_prob30[:, order1].reshape(-1, 1)
        order2 = np.argmax(self.gmm.weights_)
        d1 = gmm_prob30[:, order2].reshape(-1, 1)
        for oi in range(3):
            if oi != order1 and oi != order2:
                order3 = oi
        print("new order is, ", order1, order2, order3)
        d2 = gmm_prob30[:, order3].reshape(-1, 1)
        gmm_prob3 = np.concatenate((d0, d1), 1)
        gmm_prob3 = np.concatenate((gmm_prob3, d2), 1)
        prob0 = gmm_prob3[:, 0].reshape(-1, 1).repeat(self.train_days, axis=1)
        prob0 = prob0.reshape(len(prob0), -1, 1)
        prob1 = gmm_prob3[:, 1].reshape(-1, 1).repeat(self.train_days, axis=1)
        prob1 = prob1.reshape(len(prob1), -1, 1)
        prob2 = gmm_prob3[:, 2].reshape(-1, 1).repeat(self.train_days, axis=1)
        prob2 = prob2.reshape(len(prob2), -1, 1)
        prob = np.concatenate((prob0, prob1), 2)
        prob = np.concatenate((prob, prob2), 2)
        #         print("prob, ", prob)

        DATA = np.concatenate((DATA, prob), 2)
        print("DATA shape, ", np.array(self.DATA).shape)
        print("Label, ", np.array(Label).shape)

        dataset1 = RnnDataset(DATA, Label)
        self.train_data_loader = DataLoader(
            dataset1,
            self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            collate_fn=lambda x: x,
        )

    def refresh_dataset(self, trainX, R_X):
        self.trainX = trainX
        self.R_X = R_X
        # read sensor data to vector
        start_num = self.trainX[
            self.trainX["datetime"] == self.opt.start_point
        ].index.values[0]
        print("for sensor ", self.opt.reservoir_sensor, "start_num is: ", start_num)
        idx_num = 0  # unused?
        # foot label of train_end
        train_end = (
            self.trainX[self.trainX["datetime"] == self.opt.train_point].index.values[0]
            - start_num
        )
        print("train set length is : ", train_end)

        # the whole dataset
        k = self.trainX[self.trainX["datetime"] == self.test_end_time].index.values[0]
        f = self.trainX[self.trainX["datetime"] == self.test_start_time].index.values[
            0
        ]  # unused?
        self.sensor_data = self.trainX[start_num:k]
        self.data = np.array(self.sensor_data["value"].fillna(np.nan))
        self.diff_data = diff_order_1(self.data)
        self.data_time = np.array(self.sensor_data["datetime"].fillna(np.nan))
        self.sensor_data_norm = r_log_std_normalization_1(
            self.data, self.mean, self.std
        )  # use old mean & std
        self.sensor_data_norm1 = [[ff] for ff in self.sensor_data_norm]

        if self.opt_hinter_dim >= 1:
            # read Rain data to vector
            R_start_num = self.R_X[
                self.R_X["datetime"] == self.opt.start_point
            ].index.values[0]
            print("for sensor ", self.opt.rain_sensor, "start_num is: ", R_start_num)
            R_idx_num = 0  # unused?
            R_test_end = (
                self.R_X[self.R_X["datetime"] == self.opt.test_end].index.values[0]
                - R_start_num
            )
            print("R_X set length is : ", R_test_end)
            self.R_sensor_data = self.R_X[
                R_start_num: R_test_end + R_start_num
            ]  # e.g. 2011/7/1  22:30:00 - 2020/6/22  23:30:00
            self.R_data = np.array(self.R_sensor_data["value"].fillna(np.nan))
            self.R_data_time = np.array(self.R_sensor_data["datetime"].fillna(np.nan))
            self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(
                self.R_data
            )
            self.R_sensor_data_norm1 = [[ff] for ff in self.R_sensor_data_norm]
            gmm_input = self.R_sensor_data_norm
        else:
            gmm_input = self.sensor_data_norm

        if self.is_prob_feature == 1:

            clean_data = []
            for ii in range(len(self.sensor_data_norm)):
                if (self.sensor_data_norm[ii] is not None) and (
                    np.isnan(self.sensor_data_norm[ii]) != 1
                ):
                    clean_data.append(self.sensor_data_norm[ii])
            sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)

            data_prob3 = self.gm3.predict_proba(sensor_data_prob)
            weights3 = self.gm3.weights_
            prob_in_distribution3 = (
                data_prob3[:, 0] * weights3[0]
                + data_prob3[:, 1] * weights3[1]
                + data_prob3[:, 2] * weights3[2]
            )

            prob_like_outlier3 = 1 - prob_in_distribution3
            prob_like_outlier3 = prob_like_outlier3.reshape((len(sensor_data_prob), 1))
            print("data_prob3 shape, ", np.array(data_prob3).shape)
            recover_data = []
            temp = np.zeros(np.array(data_prob3[0]).shape)
            jj = 0
            for ii in range(len(self.sensor_data_norm)):
                if (self.sensor_data_norm[ii] is not None) and (
                    np.isnan(self.sensor_data_norm[ii]) != 1
                ):
                    recover_data.append(prob_like_outlier3[jj])
                    jj = jj + 1
                else:
                    recover_data.append(self.sensor_data_norm[ii])
            prob_like_outlier3 = np.array(recover_data, np.float32).reshape(
                len(self.sensor_data_norm), 1
            )
            self.sensor_data_norm1 = np.concatenate(
                (self.sensor_data_norm1, prob_like_outlier3), 1
            )

            # point-wise probability features, generate dim 2-5
            clean_data = []
            for ii in range(len(gmm_input)):
                if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                    clean_data.append(gmm_input[ii])
            sensor_data_prob = np.array(clean_data, np.float32).reshape(-1, 1)
            # dataset-wise gmm
            self.gmm0_means = np.squeeze(self.gmm0.means_)
            print("gmm0.means are: ", self.gmm0_means)
            print("gmm0.weights are: ", self.gmm0.weights_)
            weights3 = self.gmm0.weights_
            data_prob30 = self.gmm0.predict_proba(sensor_data_prob)
            order1 = np.argmax(weights3)
            d0 = data_prob30[:, order1].reshape(-1, 1)
            order2 = np.argmin(weights3)
            d1 = data_prob30[:, order2].reshape(-1, 1)
            for oi in range(3):
                if oi != order1 and oi != order2:
                    order3 = oi
            print("new order is, ", order1, order2, order3)
            data_prob3 = np.concatenate((d0, d1), 1)
            data_prob3 = np.concatenate(
                (data_prob3, data_prob30[:, order3].reshape(-1, 1)), 1
            )

            prob_in_distribution3 = (
                data_prob30[:, 0] * weights3[0]
                + data_prob30[:, 1] * weights3[1]
                + data_prob30[:, 2] * weights3[2]
            )

            prob_like_outlier3 = 1 - prob_in_distribution3
            prob_like_outlier3 = prob_like_outlier3.reshape(len(sensor_data_prob), 1)

            recover_data = []
            recover_prob = []
            temp = np.zeros(np.array(data_prob3[0]).shape)
            jj = 0
            for ii in range(len(gmm_input)):
                if (gmm_input[ii] is not None) and (np.isnan(gmm_input[ii]) != 1):
                    recover_data.append(prob_like_outlier3[jj])
                    recover_prob.append(data_prob3[jj])
                    jj = jj + 1
                else:
                    recover_data.append(gmm_input[ii])
                    recover_prob.append(temp)
            prob_like_outlier3 = np.array(recover_data, np.float32).reshape(
                len(gmm_input), 1
            )
            recover_prob = np.array(recover_prob, np.float32).reshape(
                len(gmm_input), -1
            )
            self.sensor_data_norm1 = np.concatenate(
                (self.sensor_data_norm1, recover_prob[:, 0:1]), 1
            )
            self.sensor_data_norm1 = np.concatenate(
                (self.sensor_data_norm1, recover_prob[:, 1:2]), 1
            )
            self.sensor_data_norm1 = np.concatenate(
                (self.sensor_data_norm1, recover_prob[:, 2:3]), 1
            )
            print("Finish prob indicator updating.")

        if self.opt_hinter_dim < 1:
            self.R_data = prob_like_outlier3.squeeze()  # diff_order_1(self.data)
            self.R_sensor_data_norm, self.R_mean, self.R_std = log_std_normalization(
                self.R_data
            )
            self.R_sensor_data_norm1 = prob_like_outlier3.squeeze()
            self.R_sensor_data_norm = self.R_sensor_data_norm1

        self.tag = gen_month_tag(self.sensor_data)  # update
        self.month, self.day, self.hour = gen_time_feature(self.sensor_data)  # update

        cos_d = cos_date(self.month, self.day, self.hour)
        cos_d = [[x] for x in cos_d]
        sin_d = sin_date(self.month, self.day, self.hour)
        sin_d = [[x] for x in sin_d]

    def gen_test_data(self):

        self.test_points = []
        self.refresh_dataset(self.trainX, self.R_X)
        print("Begin to generate test_points!")

        start_num = self.trainX[
            self.trainX["datetime"] == self.opt.start_point
        ].index.values[0]

        begin_num = (
            self.trainX[self.trainX["datetime"] == self.test_start_time].index.values[0]
            - start_num
        )
        end_num = (
            self.trainX[self.trainX["datetime"] == self.test_end_time].index.values[0]
            - start_num
        )

        iterval = self.roll

        for i in range(
            int((end_num - begin_num - self.predict_days) / iterval)
        ):  # do inference every 24 hours
            point = self.data_time[begin_num + i * iterval]
            if not np.isnan(
                np.array(
                    self.data[
                        begin_num
                        + i * iterval
                        - self.train_days: begin_num
                        + i * iterval
                        + self.predict_days
                    ]
                )
            ).any():
                self.test_points.append([point])
