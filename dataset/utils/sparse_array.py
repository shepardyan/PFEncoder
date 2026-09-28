import numpy as np
import torch
from tqdm import tqdm


def create_sparse_batch_tensor_with_predefined_size(
    data_list,
    size,
    device
) -> torch.Tensor:
    """
    Efficiently converts a list containing numpy sparse matrix data into a PyTorch 3D sparse tensor,
    using a predefined size and device.
    Args:
        data_list (List[List[np.ndarray]]): 
            A list where each element is another list `[indices_np, data_np]`.
            - indices_np: numpy array, shape (2, nnz), dtype int,
                          the first row is the row index, the second row is the col index.
            - data_np: numpy array, shape (1, nnz) or (nnz,), containing the values of non-zero elements.
        
        size (Tuple[int, int, int]): 
            The target shape of the output sparse tensor, in the format of (batch_size, num_rows, num_cols).
            
        device (torch.device, str, or None, optional):
            The target device for creating the tensor (e.g., 'cpu', 'cuda:0').
            If None, the PyTorch default device (usually 'cpu') is used.
            Defaults to None.
    Returns:
        torch.Tensor: A 3D torch.sparse_coo_tensor located on the specified device.
        
    Raises:
        ValueError: If the batch_size in 'size' does not match the length of 'data_list'.
        IndexError: If any index in the data exceeds the boundaries defined by 'size'.
    """
    # --- 1. Parameter Validation ---
    if not isinstance(size, tuple) or len(size) != 3:
        raise ValueError(f"Parameter 'size' must be a tuple containing 3 integers, but received {size}")
    batch_size, num_rows, num_cols = size
    if batch_size != len(data_list):
        raise ValueError(
            f"Batch size in 'size' ({batch_size}) does not match the length of 'data_list' ({len(data_list)})."
        )
    # Prepare empty indices and values for an empty sparse tensor, and place them on the target device
    empty_indices = torch.empty((3, 0), dtype=torch.long, device=device)
    # Determine the dtype of values. Prefer the dtype of the first non-empty item; otherwise, default to float32
    value_dtype = torch.float32
    for item in data_list:
        if item[1].size > 0:
            value_dtype = torch.from_numpy(item[1]).dtype
            break
    empty_values = torch.empty(0, dtype=value_dtype, device=device)
    # Handle the edge case of an empty list
    if not data_list:
        return torch.sparse_coo_tensor(empty_indices, empty_values, size)
    # --- 2. Data Aggregation ---
    all_indices = []
    all_values = []
    
    for batch_idx, item in enumerate(data_list):
        indices_np, data_np = item
        
        nnz = indices_np.shape[1]
        if nnz == 0:
            continue
        # [Recommended] Validate if indices are within the predefined size boundaries
        # Note: np.max on an empty array raises an error, so this check is safe after nnz > 0
        if np.max(indices_np[0]) >= num_rows or np.max(indices_np[1]) >= num_cols:
            raise IndexError(
                f"Index out of bounds found in batch {batch_idx}. "
                f"Maximum row/col index is ({np.max(indices_np[0])}, {np.max(indices_np[1])}), "
                f"but predefined size is (..., {num_rows}, {num_cols})."
            )
        # Create batch indices
        batch_indices_np = np.full((1, nnz), batch_idx, dtype=np.int64)
        
        # Stack batch, row, and col indices
        full_indices_np = np.vstack([batch_indices_np, indices_np])
        
        # Convert from numpy to torch tensor and immediately move to the target device
        all_indices.append(torch.from_numpy(full_indices_np.astype(np.int64)).to(device))
        all_values.append(torch.from_numpy(data_np.flatten()).to(device))
    # --- 3. Tensor Creation ---
    if not all_indices:
        # If nnz is 0 for all items in data_list, return an empty sparse tensor
        return torch.sparse_coo_tensor(empty_indices, empty_values, size)
        
    # Efficiently concatenate all tensor fragments (they are already on the target device)
    final_indices = torch.cat(all_indices, dim=1)
    final_values = torch.cat(all_values, dim=0)
    
    # Create sparse tensor using the predefined 'size'
    # Since indices and values are already on the target device, the resulting sparse tensor will also be on that device
    sparse_tensor = torch.sparse_coo_tensor(
        indices=final_indices,
        values=final_values,
        size=size
    )
    return sparse_tensor

class COOArray3D:
    def __init__(self, data, indices, shape):
        # First dimension is the number of 2D matrices
        self.data = data # Shape: (batch_num, nnz,)
        self.indices = indices # Shape: (3, nnz)
        self.shape = shape # (batch_num, n_rows, n_cols)
        assert len(shape) == 3, "Shape must be a 3D tuple"
    
    @classmethod
    def from_scipy_sparse(cls, sparse_matrix):
        assert sparse_matrix.ndim == 2, "Input must be a 2D sparse matrix"
        coo = sparse_matrix.tocoo()
        data = coo.data
        row = coo.row
        col = coo.col
        shape = (1, sparse_matrix.shape[0], sparse_matrix.shape[1])
        indices = np.vstack((np.zeros_like(row), row, col)) # Shape: (3, nnz)
        return cls(data, indices, shape)
    
    @classmethod
    def from_scipy_sparse_for_batch(cls, sparse_matrix):
        assert sparse_matrix.ndim == 2, "Input must be a 2D sparse matrix"
        coo = sparse_matrix.tocoo()
        data = coo.data
        row = coo.row
        col = coo.col
        return [np.vstack([row, col]), data]
    
    def save_npz(self, file):
        np.savez_compressed(file, data=self.data, indices=self.indices, shape=self.shape)

    @classmethod
    def load_npz(cls, file):
        npz = np.load(file)
        return cls(npz['data'], npz['indices'], tuple(npz['shape']))
    
    def to_torch_coo(self, dtype=torch.float32, device='cpu'):
        indices = torch.tensor(self.indices, dtype=torch.int64, device=device)
        data = torch.tensor(self.data, dtype=dtype, device=device)
        shape = self.shape
        return torch.sparse_coo_tensor(indices, data, shape, dtype=dtype, device=device)

    @classmethod
    def from_torch_coo(cls, torch_sparse_tensor):
        assert torch_sparse_tensor.ndim == 3, "Input must be a 3D sparse tensor"
        indices = torch_sparse_tensor._indices().cpu().numpy()
        data = torch_sparse_tensor._values().cpu().numpy()
        shape = tuple(torch_sparse_tensor.shape)

        return cls(data, indices, shape)
    
    def split_batch(self, verbose=False):
        batch_num = np.max(self.indices[0, :]) + 1
        batch_data = [None for _ in range(batch_num)]
        if verbose:
            for i in tqdm(range(batch_num)):
                nnz_index = self.indices[0, :] == i
                indices = self.indices[1:, nnz_index]
                datas = self.data[nnz_index].reshape(1, -1)
                batch_data[i] = [indices, datas]
        else:
            for i in range(batch_num):
                nnz_index = self.indices[0, :] == i
                indices = self.indices[1:, nnz_index]
                datas = self.data[nnz_index].reshape(1, -1)
                batch_data[i] = [indices, datas]
        return batch_data
        
    