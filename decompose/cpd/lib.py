# use_pytorch = True
use_pytorch = False


if use_pytorch:

    import torch as B
    import torch as Bla

    backend = "pytorch"
    device = "cpu"
    ##B.set_num_threads(4)
    # device = "cuda"

    def is_tensor(obj):
        return B.is_tensor(obj)

    def to_tensor(obj, dtype=None):
        return B.as_tensor(obj, dtype=dtype, device=device)

    def copy_tensor(obj, dtype=None):
        return B.tensor(obj, dtype=dtype, device=device)

    def empty(size, dtype=None):
        return B.empty(size, device=device, dtype=dtype)

    def ones(size, dtype=None):
        return B.ones(size, device=device, dtype=dtype)

    def to_numpy(obj):
        if obj.is_cuda: obj = obj.cpu()
        return obj.numpy()

    def size(obj):
        return obj.size()

    def solve(A, B):
        return Bla.solve(B, A)[0]

    def transpose(obj, axis):
        return obj.permute(tuple(axis))

    def eig(obj):
        return B.symeig(obj, True)

    def mmax(obj):
        val = B.max(B.abs(obj), 0).values
        return B.max(val, B.ones_like(val))

    def amax(obj):
        return B.max(obj, B.ones_like(obj)*1e-10)

    def pn(obj):
        if obj.is_cuda: obj = obj.cpu()
        return obj.numpy() 

    def save(obj, file, fmt=".pt"):
        B.save(obj, file+fmt)

else:
    import numpy as B
    import numpy.linalg as Bla

    backend = "numpy"

    def is_tensor(obj):
        return type(obj).__module__ == B.__name__

    def to_tensor(obj, dtype=None):
        return B.asarray(obj, dtype=dtype)

    def copy_tensor(obj, dtype=None):
        return B.array(obj, dtype=dtype)

    def empty(size, dtype=None):
        return B.empty(size, dtype=dtype)

    def ones(size, dtype=None):
        return B.ones(size, dtype=dtype)

    def to_numpy(obj):
        return obj

    def size(obj):
        return obj.shape

    def solve(A, B):
        return Bla.solve(A, B)

    def transpose(obj, axis):
        return obj.transpose(tuple(axis))

    def eig(obj):
        return Bla.eig(obj)

    def mmax(obj):
        return B.amax(B.max(B.abs(obj), axis=0)[None, :], axis=0, initial=1)

    def amax(obj):
        return B.amax(obj[None, :], axis=0, initial=1e-10)

    def pn(obj):
        return obj
    
    def save(obj, file, fmt=".npy"):
        B.save(file+fmt, obj)
