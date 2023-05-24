import torch

def _batch_mm(matrix, matrix_batch):
    """
    :param matrix: Sparse or dense matrix, size (m, n).
    :param matrix_batch: Batched dense matrices, size (b, n, k).
    :return: The batched matrix-matrix product, size (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return torch.sparse.mm(matrix, vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)

y = torch.randn(4, 4)
x = torch.randn(2, 4, 1)
sparse_y = y.to_sparse()

out1 = _batch_mm(sparse_y, x)
out2 = torch.matmul(y, x)

print(out1.shape)
print(out2.shape)
print(torch.allclose(out1, out2))
print(out1)
print(out2)

print('#############################################')

out1 = sparse_y * x
out2 = y * x

print(out1.shape)
print(out2.shape)
print(out1)
print(out2)


