import numpy as np
import torch
from sklearn import preprocessing
import pandas as pd
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_abcd_data(cfg: DictConfig):

    # ts_data = np.load(cfg.dataset.time_seires, allow_pickle=True)
    # pearson_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
    # label_df = pd.read_csv(cfg.dataset.label)

    # with open(cfg.dataset.node_id, 'r') as f:
    #     lines = f.readlines()
    #     pearson_id = [line[:-1] for line in lines]

    # with open(cfg.dataset.seires_id, 'r') as f:
    #     lines = f.readlines()
    #     ts_id = [line[:-1] for line in lines]

    # id2pearson = dict(zip(pearson_id, pearson_data))

    # id2gender = dict(zip(label_df['id'], label_df['sex']))

    # final_timeseires, final_label, final_pearson = [], [], []

    # for ts, l in zip(ts_data, ts_id):
    #     if l in id2gender and l in id2pearson:
    #         if np.any(np.isnan(id2pearson[l])) == False:
    #             final_timeseires.append(ts)
    #             final_label.append(id2gender[l])
    #             final_pearson.append(id2pearson[l])

    # encoder = preprocessing.LabelEncoder()

    # encoder.fit(label_df["sex"])

    # labels = encoder.transform(final_label)

    # scaler = StandardScaler(mean=np.mean(
    #     final_timeseires), std=np.std(final_timeseires))

    # final_timeseires = scaler.transform(final_timeseires)

    # final_timeseires, final_pearson, labels = [np.array(
    #     data) for data in (final_timeseires, final_pearson, labels)]

    # final_timeseires, final_pearson, labels = [torch.from_numpy(
    #     data).float() for data in (final_timeseires, final_pearson, labels)]


    roi_data = np.load(cfg.dataset.node_feature, allow_pickle=True)
    
    pearson_data = roi_data[:,2:].reshape(-1,180,180)
    subj_ids = roi_data[:,1]

    meta_data = pd.read_csv(cfg.dataset.label)[['subjectkey','sex']].dropna()
    subj_ids_with_label = [subj_id in meta_data['subjectkey'].values for subj_id in subj_ids]
    #subj_ids[subj_ids_with_label]
    pearson_data = np.nan_to_num(pearson_data.astype(float))
    final_pearson = pearson_data[subj_ids_with_label,:,:]

    labels = []
    for subj_id in subj_ids[subj_ids_with_label]:
        labels.append(meta_data[meta_data['subjectkey']==subj_id]['sex'].values[0])
    labels = np.array(labels)

    # dummy timeseries data, bnt does not use timeseries data   
    final_timeseries = np.zeros((final_pearson.shape[0],300))

    # final_pearson = torch.from_numpy(final_pearson.astype(float)).double()
    # labels = torch.from_numpy(labels)
    # final_timeseries = torch.zeros(final_pearson.shape[0],300)

    
    final_timeseries, final_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseries, final_pearson, labels)]

    with open_dict(cfg):
        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseries.shape[1]
    return final_timeseries, final_pearson, labels
