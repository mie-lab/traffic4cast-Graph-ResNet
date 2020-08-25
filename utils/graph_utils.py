import os
import pickle
import numpy as np
import torch
import networkx as nx
from scipy import sparse
import scipy
from scipy.sparse import coo_matrix


def csr_to_torch(A_csr):
    """transforms scipy sparse CSR matrix to a sparse torch matrix
    
    Args:
        A_csr (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    A_coo = coo_matrix(A_csr)
    values = A_coo.data
    indices = np.vstack((A_coo.row, A_coo.col))

    indices = indices.astype(np.int64)
    i = torch.LongTensor(indices)

    v = torch.FloatTensor(values.astype('float'))
    shape = A_coo.shape
    A_torch = torch.sparse.IntTensor(i, v, torch.Size(shape))

    return A_torch


def coo_to_torch(A_coo):
    """Transforms scipy sparse COO matrix to sparse torch matrix
    
    Args:
        A_coo (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    values = A_coo.data
    indices = np.vstack((A_coo.row, A_coo.col))

    indices = indices.astype(np.int64)
    i = torch.LongTensor(indices)

    v = torch.FloatTensor(values.astype('float'))
    shape = A_coo.shape
    A_torch = torch.sparse.IntTensor(i, v, torch.Size(shape))

    return A_torch


def image_to_vector(image, nn_ixs):
    """Transforms a (sparse) image into a vector representation.
    
    
    Args:
        image (TYPE): A stack of images. The last two dimensions have to be the 2D image dimensions. 
                    All other dimensions are preserved
        nn_ixs (TYPE): Indices of non-zero elements (these elements will be preserved in the vector representation)
    
    Returns:
        TYPE: Description
    """
    vec = image[..., nn_ixs[0], nn_ixs[1]]

    return vec


def vector_to_image(vec, nn_ixs, n_feat=36, batch_size=1, n=495, m=436):
    """Backtransformation of the `image_to_vector` function.
    Number of elements has to fit for an image of dimensions [batch_size, n_feat, n, m]
    
    Args:
        n:
        m:
        vec (TYPE): Description
        nn_ixs (TYPE): Description
        n_feat (int, optional): Description
        batch_size (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    zero_image = np.zeros((batch_size, n_feat, n, m))
    zero_image[..., nn_ixs[0], nn_ixs[1]] = vec

    return zero_image


def create_adj_matrix(city='Berlin', mask_threshold=10, do_subsample=None):
    """Create an adjacency matrix using a binary mask.
    Loads a representation of the traffic intensity in a city (sum over all traffic images)
    and creates a binary mask from it using a threshold.
    We create a networkx grid graph using `get_grid_graph` and delete all nodes that are zero in the 
    binary mask. 
    
    Args:
        do_subsample:
        city (str, optional): can be 'Berlin', 'Istanbul' or 'Moscow'
        mask_threshold (int, optional): A high threshold (> 100'000) only considers very active streets in the street graph
                                        A low threshold (< 100) considers all streets and some noise points in the street graph
    
    Returns:
        TYPE: Description
    """
    # create matrix
    mask_dict = pickle.load(open(os.path.join('.', 'utils', 'masks.dict'), "rb"))

    if mask_threshold > 0:
        sum_city = mask_dict[city]['sum']
        mask = sum_city > mask_threshold

    else:
        mask = mask_dict[city]['mask']

    if do_subsample is not None:
        i, j = do_subsample
        mask = mask[i:j, i:j]

    nn_ixs = np.where(mask)

    # make matrix
    mask_inv = ~mask
    to_delete = np.where(mask_inv)
    to_delete = list(zip(to_delete[0], to_delete[1]))

    m, n = mask.shape
    G = get_grid_graph(m, n)

    G.remove_nodes_from(to_delete)
    A = nx.to_scipy_sparse_matrix(G)
    A = A + scipy.sparse.identity(A.shape[0], dtype='bool', format='csr')
    return A, nn_ixs, G, mask


def get_grid_graph(n, m):
    """ Create a networkx grid graph and add edges to diagonal neighbors 

    """
    G = nx.grid_2d_graph(n, m)
    rows = range(n)
    columns = range(m)
    G.add_edges_from(((i, j), (i - 1, j - 1)) for i in rows for j in columns if i > 0 and j > 0)
    G.add_edges_from(((i, j), (i - 1, j + 1)) for i in rows for j in columns if i > 0 and j < max(columns))

    return G


def transform_shape_train(data, batch_size, n_channels=36):
    """Summary
    
    Args:
        data (TYPE): Description
        batch_size (TYPE): Description
        n_channels (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    data = data.reshape((batch_size, n_channels, -1))
    return np.squeeze(data)


def transform_shape_test(databatch_size, n_channels=9):
    """Summary
    
    Args:
        databatch_size (TYPE): Description
        n_channels (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    data = data.reshape((batch_size, n_channels, -1))
    return np.squeeze(data)


def blockify_A(A, batch_size):
    """Stack the adjacency matrix into a block diagonal matrix
    
    Args:
        A (TYPE): Description
        batch_size (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    A_list = []
    for i in range(batch_size):
        A_list.append(A)

    adj_block = scipy.sparse.block_diag((A_list), format='csr')

    return adj_block


def blockify_data(features, target, batch_size):
    """Stack batch_size dimension of the data into the other dimensions
    (has no effect for batch_size 1)
    
    Args:
        target:
        features (TYPE): Description
        batch_size (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    n_feats = features.shape[1]
    n_classes = target.shape[1]

    # get channel dimension into first place
    features = features.permute(1, 0, 2)
    target = target.permute(1, 0, 2)

    # stack batch-size dimension
    features = features.reshape(n_feats, -1)
    target = target.reshape(n_classes, -1)

    # permute channel back into last dimension
    features = features.permute(1, 0)
    target = target.permute(1, 0)

    return features, target


def unblockify_target(target, batch_size):
    """unblock data that was stacked by `blockify_data`
    
    Args:
        target:
        batch_size (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    target_list = np.split(target, batch_size)
    return target_list


def retransform_unblockify_target(target_block, nn_ixs, batch_size, dataset, n_feat=9):
    """unblock data that was stacked by `blockify_data` and transform it into an image
    input numpy - output numpy
    
    Args:
        dataset:
        target_block (TYPE): Description
        nn_ixs (TYPE): Description
        batch_size (TYPE): Description
        n_feat (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    if hasattr(dataset, 'subsample') and dataset.subsample:
        n = np.abs(dataset.n - dataset.m)
        m = n

    elif hasattr(dataset, 'dimensions') and dataset.dimensions:
        n = dataset.n
        m = dataset.m

    else:
        n = 495
        m = 436

    target_list = unblockify_target(target_block, batch_size)
    target_vector = np.stack(target_list)
    target_vector = np.moveaxis(target_vector, 2, 1)
    target_image = vector_to_image(target_vector, nn_ixs, n_feat=n_feat, batch_size=batch_size, n=n, m=m)
    return target_image


def create_coordinate_channel(h=495, w=436, b=1):
    x_coords = np.expand_dims(np.arange(w), 0)
    x_coords = np.repeat(x_coords, h, 0)
    x_coords = np.expand_dims(x_coords, 0)
    x_coords = np.expand_dims(x_coords, 0)
    x_coords = np.repeat(x_coords, b, 0) / w

    y_coords = np.expand_dims(np.arange(h), 1)
    y_coords = np.repeat(y_coords, w, 1)
    y_coords = np.expand_dims(y_coords, 0)
    y_coords = np.expand_dims(y_coords, 0)
    y_coords = np.repeat(y_coords, b, 0) / h

    np_coords = np.asarray((np.squeeze(x_coords[0, ...]), np.squeeze(y_coords[0, ...])))

    x_coords = torch.from_numpy(x_coords).to(dtype=torch.float)
    y_coords = torch.from_numpy(y_coords).to(dtype=torch.float)
    return (x_coords, y_coords)


def create_edge_index_from_adjacency_matrix(adj):
    # adj = csr
    edge_tuple = adj.nonzero()
    edge_array = np.stack(edge_tuple)
    edge_array = edge_array.astype(np.int64)
    edge_index = torch.LongTensor(edge_array)

    return edge_index
