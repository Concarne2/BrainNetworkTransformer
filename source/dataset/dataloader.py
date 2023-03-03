import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F


def init_dataloader(cfg: DictConfig,
                    final_subj_ids: np.array,
                    final_timeseires: torch.tensor,
                    final_pearson: torch.tensor,
                    labels: torch.tensor,
                    task: str) -> List[utils.DataLoader]:
    if task == 'sex':
        labels = F.one_hot(labels.to(torch.int64))
    
    subject_order = open(cfg.dataset.split_file_path, "r").readlines()
    subject_order = [x[:-1] for x in subject_order]
    train_index = np.argmax(["train" in line for line in subject_order])
    val_index = np.argmax(["val" in line for line in subject_order])
    test_index = np.argmax(["test" in line for line in subject_order])
    train_names = subject_order[train_index + 1 : val_index]
    val_names = subject_order[val_index + 1 : test_index]
    test_names = subject_order[test_index + 1 :]
    train_idx = np.where(np.in1d(final_subj_ids, train_names))[0].tolist()
    val_idx = np.where(np.in1d(final_subj_ids, val_names))[0].tolist()
    test_idx = np.where(np.in1d(final_subj_ids, test_names))[0].tolist()
    # length = final_pearson.shape[0]
    # train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    # val_length = int(length*cfg.dataset.val_set)
    # if cfg.datasz.percentage == 1.0:
    #     test_length = length-train_length-val_length
    # else:
    #     test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    train_length = len(train_idx)
    val_length = len(val_idx)
    test_length = len(test_idx)

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    # dataset = utils.TensorDataset(
    #     final_timeseires[:train_length+val_length+test_length],
    #     final_pearson[:train_length+val_length+test_length],
    #     labels[:train_length+val_length+test_length]
    # )

    # train_dataset, val_dataset, test_dataset = utils.random_split(
    #     dataset, [train_length, val_length, test_length])

    
    train_dataset = utils.TensorDataset(
        final_timeseires[train_idx],
        final_pearson[train_idx],
        labels[train_idx]
    )
    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)
    
    val_dataset = utils.TensorDataset(
        final_timeseires[val_idx],
        final_pearson[val_idx],
        labels[val_idx]
    )
    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataset = utils.TensorDataset(
        final_timeseires[test_idx],
        final_pearson[test_idx],
        labels[test_idx]
    )
    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]


def init_stratified_dataloader(cfg: DictConfig,
                               final_timeseires: torch.tensor,
                               final_pearson: torch.tensor,
                               labels: torch.tensor,
                               stratified: np.array) -> List[utils.DataLoader]:
    labels = F.one_hot(labels.to(torch.int64))
    length = final_timeseires.shape[0]
    train_length = int(length*cfg.dataset.train_set*cfg.datasz.percentage)
    val_length = int(length*cfg.dataset.val_set)
    if cfg.datasz.percentage == 1.0:
        test_length = length-train_length-val_length
    else:
        test_length = int(length*(1-cfg.dataset.val_set-cfg.dataset.train_set))

    with open_dict(cfg):
        # total_steps, steps_per_epoch for lr schedular
        cfg.steps_per_epoch = (
            train_length - 1) // cfg.dataset.batch_size + 1
        cfg.total_steps = cfg.steps_per_epoch * cfg.training.epochs

    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_length+test_length, train_size=train_length, random_state=42)
    for train_index, test_valid_index in split.split(final_timeseires, stratified):
        final_timeseires_train, final_pearson_train, labels_train = final_timeseires[
            train_index], final_pearson[train_index], labels[train_index]
        final_timeseires_val_test, final_pearson_val_test, labels_val_test = final_timeseires[
            test_valid_index], final_pearson[test_valid_index], labels[test_valid_index]
        stratified = stratified[test_valid_index]

    split2 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_length)
    for test_index, valid_index in split2.split(final_timeseires_val_test, stratified):
        final_timeseires_test, final_pearson_test, labels_test = final_timeseires_val_test[
            test_index], final_pearson_val_test[test_index], labels_val_test[test_index]
        final_timeseires_val, final_pearson_val, labels_val = final_timeseires_val_test[
            valid_index], final_pearson_val_test[valid_index], labels_val_test[valid_index]

    train_dataset = utils.TensorDataset(
        final_timeseires_train,
        final_pearson_train,
        labels_train
    )

    val_dataset = utils.TensorDataset(
        final_timeseires_val, final_pearson_val, labels_val
    )

    test_dataset = utils.TensorDataset(
        final_timeseires_test, final_pearson_test, labels_test
    )

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=cfg.dataset.drop_last)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    test_dataloader = utils.DataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=True, drop_last=False)

    return [train_dataloader, val_dataloader, test_dataloader]
