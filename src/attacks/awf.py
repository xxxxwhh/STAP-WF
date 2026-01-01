import argparse
import os
from typing import Union

import numpy as np

from attacks.base import Attack
from attacks.modules import AWFNet
from utils.general import parse_trace, feature_transform


class AWFAttack(Attack):
    """
    AWF (Automated Website Fingerprinting) Attack implementation.
    
    Feature format: [1, 3000]
        - Single channel: Direction sequence (sign of packet sizes)
        - Fixed length: 3000 packets
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def _build_model(self):
        """Build AWF model"""
        model = AWFNet(num_classes=self.nc)
        return model

    @staticmethod
    def extract(data_path: Union[str, os.PathLike], seq_length: int = 3000) -> np.ndarray:
        """
        AWF feature extraction for a single trace.
        
        Extracts direction feature (sign of packet sizes).
        
        Args:
            data_path: Path to trace file
            seq_length: Target sequence length (default: 3000 for AWF)
            
        Returns:
            feat: [1, seq_length] numpy array
        """
        trace = parse_trace(data_path)
        
        # Extract direction feature (same as DF)
        direction_feat = feature_transform(trace, feature_type='df', seq_length=seq_length)
        
        return direction_feat
