import numpy as np
from scipy import sparse
from pypower.api import case9, case14, case30, case39, case57, case118, case300, runpf, makeYbus, loadcase, ppoption, ext2int
from collections import Counter

def pad_coo_matrix(mat, max_dim):
    # Get the shape of the original matrix
    rows, cols = mat.shape
    # Ensure max_dim is greater than any dimension of the original matrix
    if max_dim < max(rows, cols):
        raise ValueError("max_dim must be greater than any dimension of the original matrix")
    
    # If the matrix is already the target size, return directly
    if rows == max_dim and cols == max_dim:
        return mat
    
    # Create a new padded matrix using COO data
    padded_mat = sparse.coo_matrix(
        (mat.data, (mat.row, mat.col)),
        shape=(max_dim, max_dim)
    )
    
    return padded_mat

def extract_topology(ppc, max_dim=None):
    """
    Extract topology data (Z matrix) from case data with optional padding.
    
    Args:
        ppc (dict): Power system case data in PYPOWER format
        max_dim (int, optional): Maximum dimension for the Z matrix (for padding)
        
    Returns:
        numpy.ndarray: Z matrix (padded if max_dim is specified)
    """
    # Build Ybus matrix
    Ybus, _, _ = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])

    Ybus = Ybus.tocoo()
    
    # Calculate Z matrix with possible padding
    Z_matrix = calculate_z_matrix(Ybus, n_dim=max_dim)
    if max_dim:
        y_bus = pad_coo_matrix(Ybus, max_dim)
    else:
        y_bus = Ybus
    return Z_matrix, y_bus

def calculate_z_matrix(Y_matrix, n_dim=None, sparse_output=False):
    """
    Calculate the impedance matrix Z with support for rank-deficient Y matrices (disconnected networks).
    
    Args:
        Y_matrix (array_like or sparse matrix): Input admittance matrix
        n_dim (int, optional): Output dimension of Z matrix, if larger than Y matrix dimension, 
                              Z matrix will be padded in the top-left corner, rest filled with zeros
        sparse_output (bool, optional): Whether to return in sparse matrix format, default is False
        
    Returns:
        Z_matrix (array_like or sparse matrix): Calculated impedance matrix
    """
    # Handle input matrix, convert to appropriate format
    if sparse.issparse(Y_matrix):
        Y_dense = Y_matrix.toarray()
    else:
        Y_dense = np.array(Y_matrix)
    
    # Get Y matrix dimension
    y_dim = Y_dense.shape[0]
    
    # Use pseudo-inverse to calculate Z matrix, can handle rank-deficient cases
    # For disconnected networks, pseudo-inverse gives the "inverse" in the least squares sense
    Z_matrix = np.linalg.pinv(Y_dense)
    
    # If a larger output dimension is specified, pad the matrix
    if n_dim is not None and n_dim > y_dim:
        # Create a zero matrix, preserving the original data type (including complex types)
        padded_Z = np.zeros((n_dim, n_dim), dtype=Z_matrix.dtype)
        # Fill the Z matrix in the top-left corner
        padded_Z[:y_dim, :y_dim] = Z_matrix
        Z_matrix = padded_Z
    
    # Return in sparse or dense format as requested
    if sparse_output:
        return sparse.csr_matrix(Z_matrix)
    else:
        return Z_matrix
    
def create_padding_mask(a_dim, n):
    """
    Create a padding mask indicating that the square matrix a is located in the top-left corner under dimension n.
    
    Args:
        a_dim (int): Dimension of the original square matrix a (a_dim x a_dim)
        n (int): Target dimension (n x n), must be >= a_dim
        framework (str): 'numpy' or 'torch', specifies the return data type
    
    Returns:
        mask: n x n mask matrix, True/1 indicates padding data position, False/0 indicates valid position
    """
    
    if n < a_dim:
        raise ValueError(f"Target dimension n ({n}) must be greater than or equal to original dimension a_dim ({a_dim})")
    mask = np.ones((n, n), dtype=bool) & ~np.eye(n, dtype=bool)
    mask[:a_dim, :a_dim] = False
    return mask

def calculate_bus_degrees(ppc, max_dim=None):
    """
    Calculate the degree of each bus, and pad to match max_dim.

    Args:
        ppc (dict): Power system case data in PYPOWER format
        max_dim (int, optional): Maximum dimension used for padding

    Returns:
        numpy.ndarray: Bus degrees array, length is max_dim (if specified)
    """
    # Extract branch data
    branch = ppc['branch']

    # Calculate the degree of each bus

    # Extract all bus indices (starting from 0)
    from_bus = branch[:, 0].astype(int)
    to_bus = branch[:, 1].astype(int)
    all_buses = np.concatenate((from_bus, to_bus))

    bus_degree_counter = Counter(all_buses)

    # Get total number of buses
    n_buses = ppc['bus'].shape[0]

    # Create degrees array
    bus_degrees = np.zeros(n_buses, dtype=np.int32)
    for bus_idx, degree in bus_degree_counter.items():
        # Ensure bus_idx is an integer for indexing
        bus_degrees[int(bus_idx)] = degree

    # If max_dim is specified, perform padding
    if max_dim is not None and max_dim > n_buses:
        padded_degrees = np.zeros((max_dim,), dtype=np.int32)
        padded_degrees[:n_buses] = bus_degrees
        bus_degrees = padded_degrees

    return bus_degrees