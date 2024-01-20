import numpy as np
import tensorflow as tf
def split_array_into_chunks(arr, chunk_size, pad_value=0):
    """Splits an array of shape (n, ...) into chunks of shape (chunk_size, ..), padding the last chunk if necessary.
    Args:
        arr: The input array of shape (n, ..).
        chunk_size: The desired chunk size along the first dimension.
        pad_value: The value to use for padding the last chunk (default: 0).

    Returns:
        A list of chunks, each of shape (chunk_size, ...) or padded to that shape.
    """
     # make the array 2d so we don't need separate cases
    if len(arr.shape) == 1:
      arr = arr.reshape(arr.shape[0], 1)
    n, m = arr.shape
    num_chunks = int(np.ceil(n / chunk_size))  # Calculate the number of chunks needed

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, n)  # Ensure end doesn't exceed array bounds
        chunk = arr[start:end]

        if end < start + chunk_size:  # Pad the last chunk if necessary
            pad_width = ((0, chunk_size - end + start), (0, 0))  # Pad only along the first dimension
            chunk = np.pad(chunk, pad_width, mode='constant', constant_values=pad_value)
        chunk = chunk.squeeze() # remove redundant 2nd dimension if it was added
        chunks.append(chunk)

    return chunks

def dense_to_spare(dataset_dict, keys):
    """
    Converts each of the values coresponding to 'keys' in the dataset to a sparse representation.
    Should be used with dataset.map. For example
    dataset.map(dense_to_space, ['key1', 'key2'])
    """
    dataset_sparse = dataset_dict.copy()
    for key in keys:
        value = dataset_dict.get(key, None)
        if value is not None:
            dataset_sparse[key] = tf.sparse.from_dense(value)
    return dataset_sparse


def sparse_to_dense(dataset_dict, keys):
    """
    Converts each of the values corresponding to 'keys' in the dataset to a dense representation.
    Should be used with dataset.map. For example
    dataset.map(sparse_to_dense, ['key1', 'key2'])
    """

    dataset_dense = dataset_dict.copy()
    for key in keys:
        value = dataset_dict.get(key, None)
        if value is not None:
            dataset_dense[key] = tf.sparse.to_dense(value)
    return dataset_dense
