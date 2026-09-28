import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import os
import json
from typing import Tuple, Optional, Union, List, Dict
import logging
from .utils.sparse_array import create_sparse_batch_tensor_with_predefined_size
from tqdm import tqdm
import cloudpickle as pickle
from typing import Tuple, Dict, Iterator

from model.feature_layout import convert_node_features_to_mode, get_feature_layout

TOPOLOGY_KEYED_ARRAYS = frozenset(
    {
        "unique_z_matrix",
        "edge_index",
        "edge_attr",
        "edge_num_edges",
        "edge_line_id",
        "edge_from_side",
        "edge_source_tap",
    }
)

def pf_dataset_collate_fn(batch):
    (z_matrices, node_features, valid_lens, bus_types, bus_degrees, y_bus_sparses) = zip(*batch)
    
    # 1. Stack Basic Features
    z_matrices_batch = torch.stack(z_matrices) 
    node_features_batch = torch.stack(node_features)
    bus_types_batch = torch.stack(bus_types)
    bus_degrees_batch = torch.stack(bus_degrees)
    
    # 2. Dynamic Padding Mask Creation (B, N, N)
    valid_lens_batch = torch.stack(valid_lens)  # Shape: (B, 1) or (B,)
    if valid_lens_batch.dim() > 1:
        valid_lens_batch = valid_lens_batch.squeeze(-1)  # Shape: (B,)

    batch_size = node_features_batch.shape[0]
    max_nodes = node_features_batch.shape[1]
    
    # Step A: Create 1D Mask (B, N)
    # True 表示该位置是有效节点，False 表示是 Padding
    seq_range = torch.arange(max_nodes, device=node_features_batch.device).unsqueeze(0) # (1, N)
    valid_nodes_mask = seq_range < valid_lens_batch.unsqueeze(1) # (B, N)
    
    # Step B: Expand to 2D Mask (B, N, N)
    # 只有当行(i)和列(j)都是有效节点时，位置(i, j)才为 True
    # (B, N, 1) & (B, 1, N) -> (B, N, N)
    padding_masks_batch = valid_nodes_mask.unsqueeze(2) & valid_nodes_mask.unsqueeze(1)
    padding_masks_batch = ~padding_masks_batch  # 反转，True 表示 Padding 位置
    padding_masks_batch.diagonal(dim1=-2, dim2=-1).fill_(False) # 对角线设为 False (padding)
    # 3. Batch Sparse Matrices
    y_bus_sparses_batch = create_sparse_batch_tensor_with_predefined_size(
        list(y_bus_sparses), 
        (batch_size, max_nodes, max_nodes), 
        "cpu"
    )
    # 4. shift the angle based on the first ref bus
    first_ref_bus_angles = []
    for i in range(batch_size):
        # Find the first reference bus (bus type == 3)
        ref_bus_indices = (bus_types_batch[i] == 3).nonzero(as_tuple=True)[0]
        if len(ref_bus_indices) > 0:
            first_ref_bus_index = ref_bus_indices[0]
            first_ref_bus_angle = node_features_batch[i, first_ref_bus_index, -1]  # Assuming angle is the last feature
            first_ref_bus_angles.append(first_ref_bus_angle)
        else:
            first_ref_bus_angles.append(0.0)  # If no reference bus, use 0 as default
    first_ref_bus_angles = torch.stack(first_ref_bus_angles)  # Shape: (B,)
    node_features_batch[:, :, -1] -= first_ref_bus_angles.unsqueeze(1)

    return z_matrices_batch, node_features_batch, padding_masks_batch, bus_types_batch, bus_degrees_batch, y_bus_sparses_batch

class PFDataset(Dataset):
    """
    A PyTorch Map-Style Dataset for compressed power flow datasets.
    
    Compression Strategy:
    1. Z-Matrix: Decoupled into `unique_z_matrix` (library) and `z_indices` (pointers).
    2. Padding Mask: Replaced by `valid_nodes` (int count). Mask created in collate.
    """
    
    def __init__(self, data_dir: str, feature_mode: str = 'raw6'):
        """
        Args:
            data_dir (str): Directory containing 'metadata.json', .bin files, and .pkl files.
        """
        self.data_dir = data_dir
        self.feature_mode = feature_mode
        self.feature_layout = get_feature_layout(feature_mode)
        self.metadata_path = os.path.join(data_dir, 'metadata.json')
        
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")
            
        self._load_metadata()
        self._open_memmaps()
        self._open_pkls()
        logging.info(f"Initialized PFDataset from {data_dir}. Samples: {self.num_samples}")
    def _load_metadata(self):
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.num_samples = self.metadata['num_samples']
        
        # We expect specific keys in the metadata now based on the new design
        self.arrays_info = self.metadata['arrays']
    def _open_memmaps(self):
        """
        Opens memory maps. Handles the logic for:
        - Standard arrays (node_feature, bus_degrees, etc.)
        - Decoupled Z-Matrix (unique library + indices)
        """
        self.memmaps = {}
        
        # Iterate over array keys defined in metadata
        for key, info in self.arrays_info.items():
            file_path = os.path.join(self.data_dir, info['binary_file'])
            dtype = np.dtype(info['dtype'])
            
            # Determine shape
            # Topology-keyed arrays are shared libraries indexed by z_indices,
            # so their metadata shape already includes the leading library size.
            if key in TOPOLOGY_KEYED_ARRAYS:
                shape = tuple(info['shape']) 
            else:
                # Standard fields: (num_samples, ...)
                shape = (self.num_samples, *info['shape'])
            
            if os.path.exists(file_path):
                self.memmaps[key] = np.memmap(file_path, dtype=dtype, mode='r', shape=shape)
            else:
                logging.warning(f"Binary file for {key} not found at {file_path}")
    def _open_pkls(self):
        """Loads the sparse Y-matrix pickle file."""
        self.pickle_data = {}
        # Assuming Y_matrix is still a pickle file as per description
        if 'pkl_files' in self.metadata:
            for key, filename in self.metadata['pkl_files'].items():
                path = os.path.join(self.data_dir, filename)
                if key == "Y_matrix":
                    print(f"Loading generic pickle data: {filename} ...")
                    with open(path, "rb") as f:
                        self.pickle_data[key] = pickle.load(f)
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        """
        Retrieves a compressed sample and reconstructs it.
        """
        # 1. Load Direct Features (Cheap memmap reads)
        node_feature_np = self.memmaps['node_feature'][idx]
        bus_types_np = self.memmaps['bus_types'][idx]
        bus_degrees_np = self.memmaps['bus_degrees'][idx]
        
        # 2. Load Valid Length (Metadata for Mask)
        # Assuming 'valid_nodes' is a (N, 1) or (N,) array in metadata
        valid_len_np = self.memmaps['valid_nodes'][idx]
        # 3. Reconstruct Z Matrix (Lookup)
        # 'z_indices' contains the pointer to the unique matrix
        z_ptr = self.memmaps['z_indices'][idx]
        
        # Force squeeze in case z_indices is stored as (B, 1)
        if isinstance(z_ptr, np.ndarray) and z_ptr.ndim > 0:
            z_ptr = z_ptr.item()
            
        # Retrieve the actual heavy matrix from the library
        # Copy is essential to detach from memmap and allow Tensor conversion
        z_matrix_np = self.memmaps['unique_z_matrix'][z_ptr]
        # 4. Conversion to Tensors
        z_matrix = torch.from_numpy(z_matrix_np.copy().astype(np.complex64))
        node_feature = torch.from_numpy(node_feature_np.copy().astype(np.float32))
        node_feature = convert_node_features_to_mode(node_feature, self.feature_mode)
        bus_types = torch.from_numpy(bus_types_np.copy().astype(np.int32))
        bus_degrees = torch.from_numpy(bus_degrees_np.copy().astype(np.int32))
        
        # Valid length is usually a scalar, keep as tensor for stack in collate
        valid_len = torch.tensor(valid_len_np, dtype=torch.int32)
        # 5. Sparse Data
        y_bus_sparse = self.pickle_data['Y_matrix'][z_ptr]

        # Note: padding_mask is NOT returned here, it is generated in collate_fn
        return z_matrix, node_feature, valid_len, bus_types, bus_degrees, y_bus_sparse
