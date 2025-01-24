import argparse
import os
import torch
import pandas as pd
from .data_provider.DS import DS
from .models.Group_GMM5 import DAN
from .models.Inference import MCANN_I
import zipfile


class Options:

    def __init__(self):

        self.parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        self.parser.add_argument(
            "--train_seed", default=1010, help="random seed for train sampling"
        )
        self.parser.add_argument(
            "--val_seed", default=2007, help="random seed for val sampling"
        )
        self.parser.add_argument(
            "--reservoir_sensor",
            default="reservoir_stor_4007_sof24",
            help="reservoir dataset",
        )
        self.parser.add_argument(
            "--os_s", type=int, default=0, help="oversampling steps"
        )
        self.parser.add_argument(
            "--os_v", type=int, default=5, help="oversampling frequency"
        )
        self.parser.add_argument(
            "--seq_weight", type=float, default=0.3, help="sequence cluster weight"
        )
        self.parser.add_argument(
            "--batchsize", type=int, default=48, help="batch size of train data"
        )
        self.parser.add_argument("--epochs", type=int, default=1, help="train epochs")
        self.parser.add_argument(
            "--learning_rate", type=float, default=0.001, help="learning rate"
        )
        self.parser.add_argument(
            "--lradj", type=str, default="type4", help="learning rate adjustment policy"
        )

        self.parser.add_argument(
            "--train_volume", type=int, default=20000, help="train set size"
        )
        self.parser.add_argument(
            "--hidden_dim", type=int, default=512, help="hidden dim of basic layers"
        )
        self.parser.add_argument(
            "--atten_dim", type=int, default=300, help="hidden dim of attention layers"
        )
        self.parser.add_argument(
            "--layer", type=int, default=2, help="number of layers"
        )

        self.parser.add_argument(
            "--input_dim", type=int, default=1, help="input dimension"
        )
        self.parser.add_argument(
            "--output_dim", type=int, default=1, help="output dimension"
        )
        self.parser.add_argument(
            "--input_len", type=int, default=15 * 24, help="length of input vector"
        )
        self.parser.add_argument(
            "--output_len", type=int, default=24 * 3, help="length of output vector"
        )

        self.parser.add_argument(
            "--val_size", type=int, default=60, help="validation set size"
        )
        self.parser.add_argument(
            "--start_point",
            type=str,
            default="1983-07-01 23:30:00",
            help="start time of the train set",
        )
        self.parser.add_argument(
            "--train_point",
            type=str,
            default="2018-06-30 23:30:00",
            help="end time of the train set",
        )
        self.parser.add_argument(
            "--test_start",
            type=str,
            default="2018-07-01 00:30:00",
            help="start time of the test set",
        )
        self.parser.add_argument(
            "--test_end",
            type=str,
            default="2019-07-01 00:30:00",
            help="end time of the test set",
        )
        self.parser.add_argument(
            "--oversampling",
            type=str,
            default=30,
            help="ratio of training data with extreme points.",
        )

        self.parser.add_argument(
            "--gpu_id", type=int, default=0, help="gpu ids: e.g. 0. use -1 for CPU"
        )
        self.parser.add_argument(
            "--ngpu", type=int, default=1, help="number of GPUs to use"
        )

        self.parser.add_argument(
            "--model", type=str, default="4009", help="model label"
        )
        self.parser.add_argument(
            "--mode",
            type=str,
            default="train",
            help="set it to train or inference with an existing pt_file",
        )
        self.parser.add_argument(
            "--arg_file",
            type=str,
            default="",
            help=".txt file. If set, reset the default parameters defined in this file.",
        )
        self.parser.add_argument(
            "--save",
            type=int,
            default=0,
            help="1 if save the predicted file of testset, else 0",
        )
        self.parser.add_argument("--outf", default="./output", help="output folder")

        self.opt = None

    def parse(self):

        self.opt = self.parser.parse_known_args()[0]
        # import model parameters
        if self.opt.arg_file != "":
            if not os.path.exists(self.opt.arg_file):
                print("File not exists: ", self.opt.arg_file)
            else:
                self.load_parameters(self.opt.arg_file)

        torch.cuda.set_device(self.opt.gpu_id)

        args = vars(self.opt)

        self.opt.name = "%s" % (self.opt.model)
        expr_dir = os.path.join(self.opt.outf, self.opt.name, "train")
        val_dir = os.path.join(self.opt.outf, self.opt.name, "val")
        test_dir = os.path.join(self.opt.outf, self.opt.name, "test")

        if not os.path.isdir(expr_dir):
            os.makedirs(expr_dir)
        if not os.path.isdir(val_dir):
            os.makedirs(val_dir)
        if not os.path.isdir(test_dir):
            os.makedirs(test_dir)

        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            for k, v in sorted(args.items()):
                if k != "arg_file":
                    opt_file.write("%s|%s\n" % (str(k), str(v)))
        return self.opt

    def get_model(self, pt):

        pt_file = os.path.basename(pt)
        pt_dir = os.path.dirname(pt)
        self.opt = self.parser.parse_known_args()[0]
        self.opt.model = str(pt_file[:-4])
        c_dir = os.getcwd()
        print("current dir: ", c_dir)
        os.chdir(pt_dir)
        with zipfile.ZipFile(pt_file, "r") as file:
            file.extract("opt.txt")
            self.load_parameters("opt.txt")
            self.opt.mode = "inference"
        os.remove("opt.txt")
        # Load model
        model = MCANN_I(self.opt)
        model.model_load(pt_file)
        os.chdir(c_dir)
        return model

    def load_parameters(self, arg_file):
        file_name = arg_file
        print("Importing parameters from: ", arg_file, "............")
        opt_dic = {}
        with open(file_name, "r") as opt_file:
            for line in opt_file:
                value = line.strip().split("|")
                opt_dic[value[0]] = value[1]
        opt_dic["arg_file"] = ""
        opt_file.close()
        #         print(opt_dic)
        args = vars(self.opt)
        for k, v in sorted(args.items()):
            n = "self.opt." + str(k)
            val = eval(n)
            val = opt_dic[str(k)]
            if n == "self.opt.os_s":
                self.opt.os_s = int(val)
            elif n == "self.opt.os_v":
                self.opt.os_v = int(val)
            elif n == "self.opt.seq_weight":
                self.opt.seq_weight = float(val)
            elif n == "self.opt.hidden_dim":
                self.opt.hidden_dim = int(val)
            elif n == "self.opt.reservoir_sensor":
                self.opt.reservoir_sensor = str(val)
            elif n == "self.opt.atten_dim":
                self.opt.atten_dim = int(val)
            elif n == "self.opt.layer":
                self.opt.layer = int(val)
            elif n == "self.opt.r_shift":
                self.opt.r_shift = int(val)
            elif n == "self.opt.train_seed":
                self.opt.train_seed = int(val)
            elif n == "self.opt.batchsize":
                self.opt.batchsize = int(val)
            elif n == "self.opt.epochs":
                self.opt.epochs = int(val)
            elif n == "self.opt.learning_rate":
                self.opt.learning_rate = float(val)
            elif n == "self.opt.train_volume":
                self.opt.train_volume = int(val)
            elif n == "self.opt.input_len":
                self.opt.input_len = int(val)
            elif n == "self.opt.output_len":
                self.opt.output_len = int(val)
            elif n == "self.opt.oversampling":
                self.opt.oversampling = int(val)
            elif n == "self.opt.event_focus_level":
                self.opt.event_focus_level = int(val)
            elif n == "self.opt.val_size":
                self.opt.val_size = int(val)
            elif n == "self.opt.start_point":
                self.opt.start_point = str(val)
            elif n == "self.opt.train_point":
                self.opt.train_point = str(val)
            elif n == "self.opt.test_start":
                self.opt.test_start = str(val)
            elif n == "self.opt.test_end":
                self.opt.test_end = str(val)
            elif n == "self.opt.gpu_id":
                self.opt.gpu_id = int(val)
            elif n == "self.opt.model":
                self.opt.model = str(val)
            elif n == "self.opt.mode":
                self.opt.mode = str(val)
            elif n == "self.opt.save":
                self.opt.save = int(val)
            elif n == "self.opt.out_f":
                self.opt.out_f = str(val)
            elif n == "self.opt.val_seed":
                self.opt.val_seed = int(val)
            self.opt.name = self.opt.model


if __name__ == "__main__":
    opt = Options().parse()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_id)
    # data prepare
    trainX = pd.read_csv(
        "./data_provider/datasets/" + opt.reservoir_sensor + ".tsv", sep="\t"
    )
    trainX.columns = ["datetime", "value"]
    trainX.sort_values("datetime", inplace=True),
    ds = DS(opt, trainX)

    # model training
    model = DAN(opt, ds)
    model.train()

    # Inferencing, saving the result to Inference_dir
    ds.refresh_dataset(trainX)
    model.model_load()
    Inference_result = model.inference()

    # Save the model
    expr_dir = os.path.join(opt.outf, opt.name, "train")
    c_dir = os.getcwd()
    os.chdir(expr_dir)
    with zipfile.ZipFile(opt.name + ".zip", "a") as zipped_f:
        zipped_f.write("opt.txt")
        zipped_f.write("GMM.pt")
        zipped_f.write("GM3.pt")
        zipped_f.write("GMM0.pt")
        zipped_f.write("Norm.txt")
    print("Model saved in: ", expr_dir + "/" + opt.name + ".zip")
    os.remove("opt.txt")
    os.remove("GMM.pt")
    os.remove("GM3.pt")
    os.remove("GMM0.pt")
    os.remove("Norm.txt")
    os.chdir(c_dir)
