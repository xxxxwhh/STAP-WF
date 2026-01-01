import os
import random
from datetime import datetime
from functools import wraps
from time import strftime
from time import time
from typing import Tuple, Union, Callable, Optional, Any

import numpy as np
import pandas as pd
import torch


def seed_everything(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_random_seed(f: Callable):
    @wraps(f)
    def wrap(*args: Optional[Any], **kw: Optional[Any]) -> Any:
        np.random.seed(datetime.now().microsecond)
        result = f(*args, **kw)
        return result

    return wrap


def timeit(f: Callable):
    @wraps(f)
    def wrap(*args: Optional[Any], **kw: Optional[Any]) -> Any:
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return wrap


def parse_trace(fdir: str, sanity_check: bool = True, return_original_length: bool = False):
    """
    Parse a trace file based on our predefined format
    Args:
        fdir: trace file path
        sanity_check: whether to perform sanity check and preprocessing
        return_original_length: if True, return (trace, original_length) before cropping
    Returns:
        trace: np.ndarray, preprocessed trace
        original_length: int (optional), length before cropping
    """
    trace = pd.read_csv(fdir, delimiter="\t", header=None)
    trace = np.array(trace)
    original_length = len(trace)  # 记录裁剪前的长度

    if sanity_check:
        # it is possible the trace has a long tail
        # if there is a time gap between two bursts larger than CUT_OFF_THRESHOULD
        # We cut off the trace here sicne it could be a long timeout or
        # maybe the loading is already finished
        # Set a very conservative value
        CUT_OFF_THRESHOLD = 15
        start, end = 0, len(trace)
        ipt_burst = np.diff(trace[:, 0])
        ipt_outlier_inds = np.where(ipt_burst > CUT_OFF_THRESHOLD)[0]

        if len(ipt_outlier_inds) > 0:
            outlier_ind_first = ipt_outlier_inds[0]
            if outlier_ind_first < 50:
                start = outlier_ind_first + 1
            outlier_ind_last = ipt_outlier_inds[-1]
            if outlier_ind_last > 50:
                end = outlier_ind_last + 1
        trace = trace[start:end].copy()

        # remove the first few lines that are incoming packets
        start = -1
        for time, size in trace:
            start += 1
            if size > 0:
                break

        trace = trace[start:].copy()
        trace[:, 0] -= trace[0, 0]
        assert trace[0, 0] == 0
    
    if return_original_length:
        return trace, original_length
    return trace


def feature_transform(sample: np.ndarray, feature_type: str, seq_length: int) -> np.ndarray:
    """
    Transform a raw sample to the specific feature space.
    :return a numpy array of shape (1 or 2, seq_length)
    """
    if feature_type == 'df':
        feat = np.sign(sample[:, 1])

    elif feature_type == 'tiktok':
        feat = sample[:, 0] * np.sign(sample[:, 1])

    elif feature_type == 'tam':
        max_load_time = 80.0  # s
        time_window = 0.044  # s

        cut_off_time = min(max_load_time, float(sample[-1, 0]))
        num_bins = int(cut_off_time / time_window) + 1
        bins = np.linspace(0, num_bins * time_window, num_bins).tolist() + [np.inf]

        outgoing = sample[np.sign(sample[:, 1]) > 0]
        incoming = sample[np.sign(sample[:, 1]) < 0]

        cnt_outgoing, _ = np.histogram(outgoing[:, 0], bins=bins)
        cnt_incoming, _ = np.histogram(incoming[:, 0], bins=bins)

        # merge to 2d feature
        feat = np.stack((cnt_outgoing, cnt_incoming), axis=1)
        assert feat.flatten().sum() == len(sample), \
            "Sum of feature ({}) is not equal to the length of the trace ({}). BUG?".format(
                feat.flatten().sum(), len(sample))

    elif feature_type == 'burst':
        sample = sample[:, 1]
        # Create a mask for consecutive elements that are the same
        mask = np.where(np.sign(sample[:-1]) != np.sign(sample[1:]))[0] + 1
        mask = np.concatenate((mask, [len(sample)]))  # add the last index
        # Count the number of elements between sign changes
        feat = np.diff(mask, prepend=0)
        assert sum(feat) == len(sample), \
            "Sum of burst lengths ({}) is not equal to the length of the trace ({}). BUG?".format(sum(feat),
                                                                                                  len(sample))
    
    elif feature_type == 'dt2':
        # DT2 feature for VarCNN: [direction, time_diff]
        # Channel 0: direction (sign of packet sizes)
        X_dir = np.sign(sample[:, 1])
        # Channel 1: time differences (diff of absolute timestamps)
        X_time = np.abs(sample[:, 0])
        X_time = np.diff(X_time, prepend=0)
        X_time[X_time < 0] = 0  # Ensure no negative values
        # Stack as [2, N]
        feat = np.stack([X_dir, X_time], axis=0)
        # Will be transposed later to [2, seq_length]
        return feat[:, :seq_length] if feat.shape[1] >= seq_length else np.pad(feat, ((0, 0), (0, seq_length - feat.shape[1])))
    
    elif feature_type in ['taf', 'mtaf']:
        # TAF/MTAF feature for ARES: [8, seq_length]
        X_dir = np.sign(sample[:, 1])  # direction [N]
        
        # Pad/truncate to seq_length
        if len(X_dir) < seq_length:
            X_dir = np.pad(X_dir, (0, seq_length - len(X_dir)))
        else:
            X_dir = X_dir[:seq_length]
        
        # Replicate to 8 channels: [8, seq_length]
        feat = np.tile(X_dir[np.newaxis, :], (8, 1))
        return feat  # Directly return [8, seq_length], skip post-processing
    
    else:
        raise NotImplementedError("Feature type {} is not implemented.".format(feature_type))

    # make sure 2d
    if len(feat.shape) == 1:
        feat = feat[:, np.newaxis]
    # pad to seq_length
    if len(feat) < seq_length:
        pad = np.zeros((seq_length - len(feat), feat.shape[1]))
        feat = np.concatenate((feat, pad))
    feat = feat[:seq_length, :]
    return np.transpose(feat, (1, 0))


def get_flist_label(data_path: Union[str, os.PathLike], mon_cls: int, mon_inst: int, unmon_inst: int,
                    suffix: str = '.cell', mon_inst_start: int = 0, mon_inst_end: int = None,
                    include_adversarial: bool = True, adversarial_path: str = '/data_2_mnt/xuwenhai/wfzoo/adversarial_samples/traces') \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a list of file paths and corresponding labels.
    :param data_path: the path to the data directory
    :param mon_cls: number of monitored classes
    :param mon_inst: number of monitored instances per class (ignored if mon_inst_end is specified)
    :param unmon_inst: number of unmonitored instances
    :param suffix: file suffix
    :param mon_inst_start: start index for monitored instances (inclusive)
    :param mon_inst_end: end index for monitored instances (exclusive), if None uses mon_inst_start + mon_inst
    :param include_adversarial: whether to include adversarial samples in addition to original data
    :param adversarial_path: path to adversarial samples directory (e.g., './adversarial_samples/traces')
                            required if include_adversarial=True
    :return: a list of file paths and a list of corresponding labels
    """
    # Determine end index for monitored instances
    if mon_inst_end is None:
        mon_inst_end = mon_inst_start + mon_inst
    
    flist = []
    labels = []
    
    # 1. 加载原始数据
    for cls in range(mon_cls):
        for inst in range(mon_inst_start, mon_inst_end):
            pth = os.path.join(data_path, '{}-{}{}'.format(cls, inst, suffix))
            if os.path.exists(pth):
                flist.append(pth)
                labels.append(cls)
    for inst in range(unmon_inst):
        pth = os.path.join(data_path, '{}{}'.format(inst, suffix))
        if os.path.exists(pth):
            flist.append(pth)
            labels.append(mon_cls)
    
    # 2. 如果需要,追加对抗样本
    if include_adversarial:
        if adversarial_path is None:
            raise ValueError("adversarial_path must be specified when include_adversarial=True")
        
        # 对抗样本文件名格式: {label}_{global_index}.cell
        # global_index是全局递增的索引(跨类别)
        import glob
        adv_files = glob.glob(os.path.join(adversarial_path, f'*{suffix}'))
        
        adv_count = 0
        for adv_path in adv_files:
            fname = os.path.basename(adv_path)
            try:
                # 解析文件名: {label}_{global_index}.cell
                parts = fname.replace(suffix, '').split('_')
                label = int(parts[0])  # 第一部分是标签
                
                # 过滤: 只加载指定范围的类别
                if label < mon_cls:
                    flist.append(adv_path)
                    labels.append(label)
                    adv_count += 1
            except (ValueError, IndexError):
                # 文件名格式不匹配,跳过
                continue
        
        if adv_count == 0:
            raise ValueError(f"No adversarial samples found in {adversarial_path}!")

    assert len(flist) > 0, "No files found in {}!".format(data_path)
    return np.array(flist), np.array(labels)


def load_npz_dataset(npz_path: str, return_mapping: bool = False) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, dict]]:
    """
    Load dataset from NPZ file and convert string labels to numeric indices.
    Args:
        npz_path: path to the .npz file
        return_mapping: if True, also return the label mapping dict
    Returns:
        data: np.ndarray, shape (n_samples, seq_length), feature data
        labels: np.ndarray, shape (n_samples,), numeric labels (0, 1, 2, ...)
        label_mapping: dict (optional), mapping from class_name to class_index
    """
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    # Load NPZ file
    npz_data = np.load(npz_path, allow_pickle=True)
    
    # Extract data and labels
    if 'data' not in npz_data or 'labels' not in npz_data:
        raise ValueError(f"NPZ file must contain 'data' and 'labels' keys. Found: {npz_data.files}")
    
    data = npz_data['data']
    label_strings = npz_data['labels']
    
    # Build label mapping: string -> numeric index (preserve order of first appearance)
    seen = {}
    label_mapping = {}
    for label in label_strings:
        if label not in seen:
            label_mapping[label] = len(seen)
            seen[label] = True
    
    # Convert string labels to numeric indices
    labels = np.array([label_mapping[label] for label in label_strings])
    
    if return_mapping:
        return data, labels, label_mapping
    return data, labels


def build_label_mapping(label_strings: np.ndarray) -> Tuple[np.ndarray, dict]:
    """
    Build numeric labels from string labels.
    Args:
        label_strings: np.ndarray of string labels
    Returns:
        labels: np.ndarray of numeric labels
        label_mapping: dict mapping from class_name to class_index
    """
    # Build label mapping: preserve order of first appearance
    seen = {}
    label_mapping = {}
    for label in label_strings:
        if label not in seen:
            label_mapping[label] = len(seen)
            seen[label] = True
    
    labels = np.array([label_mapping[label] for label in label_strings])
    return labels, label_mapping


def init_directories(output_parent_dir: Union[str, os.PathLike], defense_name: str) -> str:
    # Create a results dir if it doesn't exist yet
    if not os.path.exists(output_parent_dir):
        os.makedirs(output_parent_dir)

    # Define output directory
    timestamp = strftime('%m%d_%H%M%S')
    output_dir = os.path.join(output_parent_dir, defense_name + '_' + timestamp)
    os.makedirs(output_dir)
    return output_dir
