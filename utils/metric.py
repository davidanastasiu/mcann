import numpy as np
from sklearn.metrics import mean_absolute_percentage_error

def RSE(pred, true):
        return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(pred - true))

def MSE(pred, true):
    return np.mean((pred - true) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    pred = np.squeeze(pred)
    true = np.squeeze(true)
    return mean_absolute_percentage_error(np.array(true)+1, np.array(pred)+1)


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(model, pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)

    return mae, mse, rmse, mape

def metric_g(name, pre, gt, inter=72):
    pre = np.array(pre)
    gt = np.array(gt)
    ll = int(len(pre)/inter)
    mae_all = []
    mse_all = []
    rmse_all = []
    mape_all = []

    l2 = []
    l3 = []
    lll=[]
    for i in range(ll):
        mae, mse, rmse, mape = metric(name, pre[i*inter:(i+1)*inter], gt[i*inter:(i+1)*inter])
        rmse_all.append(rmse)
        mape_all.append(mape)

    l2.append(np.around(np.mean(np.array(rmse_all)),2))
    l3.append(np.around(np.mean(np.array(mape_all)),3))
    lll.append(l2)
    lll.append(l3)

    return lll

def metric_rolling1(name, pre, gt, rm=16, inter=72):
    pre = np.array(pre)
    gt = np.array(gt)
    ll = int(len(pre)/72)
    mae_all = []
    mse_all = []
    rmse_all = []
    mape_all = []
    l2 = []
    l3 = []
    lll=[]
    for i in range(ll):
        mae, mse, rmse, mape = metric(name, pre[i*inter:(i*inter+rm)], gt[i*inter:(i*inter+rm)])
        rmse_all.append(rmse)
        mape_all.append(mape)
    l2.append(np.around(np.mean(np.array(rmse_all)),2))
    l3.append(np.around(np.mean(np.array(mape_all)),3))
    lll.append(l2)
    lll.append(l3)
    print("For rolling prediction: ", name)
    print("RMSE: ", np.array(lll[0][0]))
    print("MAPE: ", np.array(lll[1][0]))
    return lll

def metric_rolling(name, pre, gt, rm=16, inter=72):
    pre = np.array(pre)
    gt = np.array(gt)
    ll = int(len(pre)/inter)
    pre_all = []
    gt_all = []
    for i in range(ll):
        pre_all.extend(pre[i*inter:(i*inter+rm)])
        gt_all.extend(gt[i*inter:(i*inter+rm)])
    _, _, rmse, mape = metric(name, np.array(pre_all), np.array(gt_all))

    print("For rolling prediction: ", name)
    print("RMSE: ", rmse)
    print("MAPE: ", mape)
    return rmse, mape