import argparse
import os
from typing import Union

import numpy as np

from attacks.base import Attack
from attacks.modules import VarCNNNet
from utils.general import parse_trace, feature_transform


class VarCNNAttack(Attack):
    """
    VarCNN Attack implementation.
    Uses dual-encoder architecture for direction and timing features.
    
    Feature format: [2, seq_length] (DT2 method)
        - Channel 0: Direction sequence (sign of packet sizes)
        - Channel 1: Time differences (diff of timestamps)
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def _build_model(self):
        """Build VarCNN model"""
        model = VarCNNNet(num_classes=self.nc)
        return model

    @staticmethod
    def extract(data_path: Union[str, os.PathLike], seq_length: int) -> np.ndarray:
        """
        VarCNN feature extraction for a single trace.
        
        Extracts two features (DT2 method):
        1. Direction: sign of packet sizes (±1)
        2. Time differences: diff of timestamps
        
        Args:
            data_path: Path to trace file
            seq_length: Target sequence length
            
        Returns:
            feat: [2, seq_length] numpy array
        """
        trace = parse_trace(data_path)
        
        # Extract DT2 feature: [direction, time_diff]
        feat = feature_transform(trace, feature_type='dt2', seq_length=seq_length)
        
        return feat
