import argparse
import os
from typing import Union

import numpy as np

from attacks.base import Attack
from attacks.modules import ARESNet
from utils.general import parse_trace, feature_transform


class ARESAttack(Attack):
    """
    ARES (Attention-based Residual nEtwork for website fingerprinting) Attack implementation.
    
    Feature format: [8, seq_length]
        - 8 channels: TAF (Time-Aware Features) or MTAF (Multi-scale TAF)
        - Default seq_length: 8000
    """
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

    def _build_model(self):
        """Build ARES model"""
        model = ARESNet(num_classes=self.nc)
        return model

    @staticmethod
    def extract(data_path: Union[str, os.PathLike], seq_length: int = 8000) -> np.ndarray:
        """
        ARES feature extraction for a single trace.
        
        Extracts TAF/MTAF features (8 channels).
        
        Args:
            data_path: Path to trace file
            seq_length: Target sequence length (default: 8000 for ARES)
            
        Returns:
            feat: [8, seq_length] numpy array
        """
        trace = parse_trace(data_path)
        
        # Extract TAF feature: [8, seq_length]
        feat = feature_transform(trace, feature_type='taf', seq_length=seq_length)
        
        return feat
