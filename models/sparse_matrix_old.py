import torch
from pdb import set_trace

from mesa import packbit

class SparseTensor(object):
    def __init__(self, tensor, mask):
        self.shape = tensor.shape
        mask = mask.reshape(-1)
        self.sparse = tensor.reshape(-1)[mask]
        self.mask = packbit.packbits_padded(mask)

    def to_dense(self, ):
        mask = packbit.unpackbits_padded(self.mask).to(dtype=torch.bool)
        self.dense = torch.zeros(mask.shape, device=self.sparse.device, dtype=self.sparse.dtype)
        self.dense[mask] = self.sparse
        return self.dense.reshape(self.shape)


if __name__ == "__main__":
    with torch.no_grad():
        attn = torch.rand(64, 12, 196, 196)
        # attn = torch.rand(1, 1, 6, 6)
        mask = attn > 0.5
        attn_sparse = mask * attn

        attn_sparse_save = SparseTensor(attn_sparse, mask)
        attn_sparse_2 = attn_sparse_save.to_dense()

        print("torch.norm(attn_sparse-attn_sparse_2) is {}".format(torch.norm(attn_sparse-attn_sparse_2)))
