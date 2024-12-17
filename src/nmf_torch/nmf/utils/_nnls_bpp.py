### NNLS Block principal pivoting: Kim and Park et al 2011.

import numpy as np
import torch
# import pyximport
# pyximport.install(setup_args={"script_args" : ["--verbose"]})
from src.nmf_torch.nmf.cylib.nnls_bpp_utils import _nnls_bpp

def nnls_bpp(CTC, CTB, X, device_type) -> int:
    """
        min ||CX-B||_F^2
        KKT conditions:
           1) Y = CTC @ X - CTB
           2) Y >= 0, X >= 0
           3) XY = 0
        Return niter; if niter < 0, nnls_bpp does not converge.
    """
    # CTC = C.T @ C
    # CTB = C.T @ B

    # dtype = C.dtype
    # X = torch.zeros((q, r), dtype=dtype)
    if isinstance(CTC, torch.Tensor):
        CTC = CTC.numpy()
    if isinstance(CTB, torch.Tensor):
        CTB = CTB.numpy()
    if isinstance(X, torch.Tensor):
        X = X.numpy()

    return _nnls_bpp(CTC, CTB, X, device_type)


    # if device_type == 'cpu':
    #     return _nnls_bpp(CTC, CTB, X, 'cpu')
    # else:
    #     # X_cpu = torch.zeros_like(X, device='cpu')
    #     n_iter = _nnls_bpp(CTC.cpu().numpy(), CTB.cpu().numpy(), X.numpy(), 'gpu')
    #     X[:] = X_cpu.cuda()
    #     return n_iter


if __name__ == "__main__":
    # CX - B
    # B = Dx1, C = DxK, X = Kx1
    import time

    torch.manual_seed(0)

    num_samples = 512
    B = torch.rand((64, num_samples)) ** 2
    C = torch.rand((64, 10)) ** 2
    X = torch.zeros((10, num_samples))

    CTC = C.T @ C
    CTB = C.T @ B

    st = time.time()
    n_iters = nnls_bpp(CTC, CTB, X, 'cuda')
    print('torch nmf Elapsed time:', time.time() - st)
    # print('X', X)

    from scipy.optimize import nnls

    st = time.time()
    for i in range(num_samples):
        X2, _ = nnls(C, B[:, i])
    print('scipy Elapsed time:', time.time() - st)
    # print('X2', torch.FloatTensor(X2).unsqueeze(1))


    print('Trying with real data')
    test_concept_weights = np.load('/home/nkondapa/PycharmProjects/ConceptBook/test_concept_weights.npy')
    test_activations = np.load('/home/nkondapa/PycharmProjects/ConceptBook/test_activations.npy')

    C = torch.FloatTensor(test_concept_weights)
    B = torch.FloatTensor(test_activations)
    X = torch.zeros((C.shape[1], B.shape[1]))

    st = time.time()
    CTC = C.T @ C
    CTB = C.T @ B
    print(time.time() - st)
    n_iters = nnls_bpp(CTC, CTB, X, 'cpu')
    print('torch nmf Elapsed time:', time.time() - st)
    print(n_iters)
    st = time.time()
    x2s = []
    rnorms = []
    for i in range(num_samples):
        X2, rnorm = nnls(C, B[:, i])
        x2s.append(X2)
        rnorms.append(rnorm)
    print('scipy Elapsed time:', time.time() - st)
    X2 = torch.FloatTensor(np.stack(x2s)).T
    rnorms = np.array(rnorms)

    # print(X)
    # print(X2)

    rnormx1 = torch.norm(C @ X - B, dim=0)
    rnormx2 = torch.norm(C @ X2 - B, dim=0)


    # print('rnorms', rnorms[:10])
    # print('rnormx1', rnormx1[:10])
    # print('rnormx2', rnormx2[:10])

    # print(np.mean(rnorms), torch.mean(rnormx1), torch.mean(rnormx2))