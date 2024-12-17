from sklearn.decomposition import NMF, PCA
from src.nmf_torch.nmf.nmf_models import NMFBatchMU, NMFBatchHALS, NMFBatchNnlsBpp, NMFOnlineMU, NMFOnlineHALS, \
    NMFOnlineNnlsBpp
from sklearn.cluster import KMeans
from pymf.pymf.cnmf import CNMF
from pymf.pymf.snmf import SNMF
from pymf.pymf.nmf import NMF as PyMFNMF
from src.nmf_torch.nmf.utils._nnls_bpp import nnls_bpp
from typing import Union, Dict
import numpy as np
import torch
from functools import partial
import pickle as pkl
from src.fnnls.src import fnnls


class DictionaryLearner:

    def __init__(self, method: str = 'nmf', params: dict = None):
        self.method = method
        self.params = params
        self.reducer = self.build_reducer(self.method, self.params)

    @staticmethod
    def build_reducer(method, params):
        num_components = params.get('num_concepts', 10)
        if method == 'nmf':
            seed = params.get('seed', None)
            init = 'random' if seed is None else 'nndsvd'
            return NMF(n_components=num_components, random_state=seed, init=init)
        elif method == 'pymf-nmf':
            return partial(PyMFNMF, num_bases=num_components)
        elif method == 'torch-nmf':
            return TorchNMFWrapper(n_components=num_components)
        elif method == 'pca':
            return PCA(n_components=num_components)
        elif method == 'kmeans':
            return KMeans(n_clusters=num_components)
        elif method == 'cnmf':
            return partial(CNMF, num_bases=num_components)
        elif method == 'snmf':
            return partial(SNMF, num_bases=num_components)
        else:
            raise ValueError(f'Unknown method {method}')

    def transform(self, X, W=None, params=None):
        if self.method == 'nmf':
            U = self.reducer.transform(X)
            W = self.reducer.components_
            err = self.reducer.reconstruction_err_
            out = {'U': U, 'W': W, 'err': err}
        elif self.method == 'pymf-nmf':
            niters = params['n_iters']
            nmf_mdl = self.reducer
            nmf_mdl.W = W
            del nmf_mdl.H
            nmf_mdl.factorize(niter=niters, compute_w=False)
            W = nmf_mdl.W
            U = nmf_mdl.H
            err = nmf_mdl.ferr
            out = {'W': W, 'U': U, 'err': err}
        elif self.method == 'torch-nmf':
            U = self.reducer.transform(X, W)
            out = {'U': U, 'W': W, 'err': 0}
        elif self.method == 'pca':
            U = self.reducer.transform(X)
            W = self.reducer.components_
            err = np.linalg.norm(X - U @ W)
            out = {'U': U, 'W': W, 'err': err, 'mean': self.reducer.mean_,
                   'explained_variance': self.reducer.explained_variance_, 'whiten': self.reducer.whiten}
        elif self.method == 'cnmf':
            G = params['G']
            n_iters = params['n_iters']
            cnmf_mdl = self.reducer

            cnmf_mdl.W = W.T
            cnmf_mdl.G = G
            del cnmf_mdl.H

            cnmf_mdl.factorize(compute_w=False, niter=n_iters)
            W = cnmf_mdl.W
            U = cnmf_mdl.H
            G = cnmf_mdl.G
            err = cnmf_mdl.ferr
            out = {'W': W.T, 'U': U.T, 'G': G, 'err': err}
        elif self.method == 'snmf':
            snmf_mdl = self.reducer
            snmf_mdl.W = W.T
            snmf_mdl.factorize(niter=10, compute_w=False)
            W = snmf_mdl.W
            U = snmf_mdl.H
            err = snmf_mdl.ferr
            out = {'W': W.T, 'U': U.T, 'err':  err}
        else:
            raise NotImplementedError
        return out

    def fit_transform(self, X):
        if self.method == 'nmf':
            U = self.reducer.fit_transform(X)  # expects data in shape (num_samples, num_features)
            W = self.reducer.components_
            err = self.reducer.reconstruction_err_
            out = {'U': U, 'W': W, 'err': err}
        elif self.method == 'pymf-nmf':
            nmf_mdl = self.reducer(X.T)
            self.reducer = nmf_mdl
            nmf_mdl.factorize(niter=20)
            W = nmf_mdl.W
            U = nmf_mdl.H
            err = nmf_mdl.ferr
            out = {'W': W, 'U': U, 'err': err}
        elif self.method == 'torch-nmf':
            U = self.reducer.fit_transform(X)
            W = self.reducer.model.W
            out = {'W': W, 'U': U, 'err': 0}

        elif self.method == 'pca':
            U = self.reducer.fit_transform(X)  # expects data in shape (num_samples, num_features)
            W = self.reducer.components_
            err = np.linalg.norm(X - U @ W)
            out = {'U': U, 'W': W, 'err': err, 'mean': self.reducer.mean_,
                   'explained_variance': self.reducer.explained_variance_, 'whiten': self.reducer.whiten}
        elif self.method == 'cnmf':
            if isinstance(X, torch.Tensor):
                X = X.numpy() # convert torch to numpy
            cnmf_mdl = self.reducer(X.T)  # expects data in shape (num_features, num_samples)
            self.reducer = cnmf_mdl
            cnmf_mdl.factorize(niter=10)
            W = cnmf_mdl.W
            U = cnmf_mdl.H
            G = cnmf_mdl.G
            err = cnmf_mdl.ferr
            out = {'W': W.T, 'U': U.T, 'G': G, 'err': err}
        elif self.method == 'snmf':
            if isinstance(X, torch.Tensor):
                X = X.numpy()
            snmf_mdl = self.reducer(X.T)
            self.reducer = snmf_mdl
            snmf_mdl.factorize(niter=10)
            W = snmf_mdl.W
            U = snmf_mdl.H
            err = snmf_mdl.ferr
            out = {'W': W.T, 'U': U.T, 'err': err}
        else:
            raise NotImplementedError

        return out

    @staticmethod
    def static_transform(method, X, W, params=None):

        if method == 'nmf':
            num_components = W.shape[0]
            reducer = DictionaryLearner.build_reducer(method, {'num_concepts': num_components})
            U = reducer.fit_transform(X, H=W) # scikit-learn uses different naming convention (H = W)
            err = reducer.reconstruction_err_
            out = {'U': U, 'W': W, 'err': err}

        elif method == 'pymf-nmf':
            num_components = W.shape[1]
            reducer = DictionaryLearner.build_reducer(method, {'num_concepts': num_components})
            nmf_mdl = reducer(X.T)
            nmf_mdl.W = W
            n_iters = params['n_iters']
            nmf_mdl.factorize(niter=n_iters, compute_w=False)
            W = nmf_mdl.W
            U = nmf_mdl.H
            err = nmf_mdl.ferr
            out = {'W': W, 'U': U, 'err': err}
        elif method == 'torch-nmf':
            skip_err_comp = params.get('skip_err_comp', False)
            U = TorchNMFWrapper.transform(X, W, params.get('device', 'cpu'))
            if not skip_err_comp:
                err = torch.linalg.vector_norm((torch.FloatTensor(X) - (U @ W)), dim=(0, 1))
            else:
                err = -1
            out = {'U': U, 'W': W, 'err': err}

        elif method == 'pca':
            num_components = W.shape[0]
            reducer = DictionaryLearner.build_reducer(method, {'num_concepts': num_components})
            reducer.components_ = W
            reducer.mean_ = params.get('mean', None)
            reducer.explained_variance_ = params.get('explained_variance', None)
            reducer.whiten = params.get('whiten', False)
            U = reducer.transform(X)
            err = np.linalg.norm(X - U @ W)
            out = {'U': U, 'W': W, 'err': err}
        elif method == 'cnmf':
            U = np.stack([fnnls.fnnls(W.T, X[i])[0] for i in range(X.shape[0])])
            err = np.linalg.norm(X - U @ W)
            out = {'U': U, 'W': W, 'err': err}
        elif method == 'snmf':
            U = np.stack([fnnls.fnnls(W.T, X[i])[0] for i in range(X.shape[0])])
            err = np.linalg.norm(X - U @ W)
            out = {'U': U, 'W': W, 'err': err}
        elif method == 'fnnls':
            absolute_max_iters = params.get('absolute_max_iters', 200)
            skip_err_comp = params.get('skip_err_comp', False)
            # st = time.time()
            U = np.stack([fnnls.fnnls(W.T, X[i], absolute_max_iters=absolute_max_iters)[0] for i in range(X.shape[0])])
            # print(f'No mp Elapsed time: {time.time() - st:.2f} seconds.')
            # st = time.time()
            # args = [(W.T, X[i], absolute_max_iters) for i in range(X.shape[0])]
            # with multiprocessing.get_context('spawn').Pool(8) as pool:
            #     u_stack = pool.starmap(DictionaryLearner.fnnls_wrapper, args)
            # U = np.stack(u_stack)
            # print(f'MP Elapsed time: {time.time() - st:.2f} seconds.')

            # u_stack = []
            # for i in range(X.shape[0]):
            #     st = time.time()
            #     u = fnnls.fnnls(W.T, X[i], absolute_max_iters=absolute_max_iters)[0]
            #     print(f'Elapsed time: {time.time() - st:.2f} seconds.')
            #     u_stack.append(u)

            if not skip_err_comp:
                err = np.linalg.norm(X - U @ W)
            else:
                err = -1
            out = {'U': U, 'W': W, 'err': err}
        else:
            raise NotImplementedError
        return out

    @staticmethod
    def fnnls_wrapper(WT, x, absolute_max_iters):
        return fnnls.fnnls(WT, x, absolute_max_iters=absolute_max_iters)[0]

    def inverse_transform(self, X):
        raise NotImplementedError

    def save(self, out: Dict, path: str):
        with open(path, 'wb') as f:
            pkl.dump(out, f)


class TorchNMFWrapper:

    def __init__(self,
                 n_components: int,
                 init: str = "nndsvdar",
                 beta_loss: Union[str, float] = "frobenius",
                 algo: str = "halsvar",
                 mode: str = "batch",
                 tol: float = 1e-4,
                 n_jobs: int = -1,
                 random_state: int = 0,
                 use_gpu: bool = False,
                 alpha_W: float = 0.0,
                 l1_ratio_W: float = 0.0,
                 alpha_H: float = 0.0,
                 l1_ratio_H: float = 0.0,
                 fp_precision: Union[str, torch.dtype] = "float",
                 batch_max_iter: int = 500,
                 batch_hals_tol: float = 0.05,
                 batch_hals_max_iter: int = 200,
                 online_max_pass: int = 20,
                 online_chunk_size: int = 5000,
                 online_chunk_max_iter: int = 200,
                 online_h_tol: float = 0.05,
                 online_w_tol: float = 0.05,
                 ):
        self.n_components = n_components
        self.init = init
        self.beta_loss = beta_loss
        self.algo = algo
        self.mode = mode
        self.tol = tol
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.alpha_W = alpha_W
        self.l1_ratio_W = l1_ratio_W
        self.alpha_H = alpha_H
        self.l1_ratio_H = l1_ratio_H
        self.fp_precision = fp_precision
        self.batch_max_iter = batch_max_iter
        self.batch_hals_tol = batch_hals_tol
        self.batch_hals_max_iter = batch_hals_max_iter
        self.online_max_pass = online_max_pass
        self.online_chunk_size = online_chunk_size
        self.online_chunk_max_iter = online_chunk_max_iter
        self.online_h_tol = online_h_tol
        self.online_w_tol = online_w_tol
        if beta_loss == 'frobenius':
            beta_loss = 2
        elif beta_loss == 'kullback-leibler':
            beta_loss = 1
        elif beta_loss == 'itakura-saito':
            beta_loss = 0
        elif not (isinstance(beta_loss, int) or isinstance(beta_loss, float)):
            raise ValueError(
                "beta_loss must be a valid value: either from ['frobenius', 'kullback-leibler', 'itakura-saito'], or a numeric value.")

        device_type = 'cpu'
        if use_gpu:
            if torch.cuda.is_available():
                device_type = 'cuda'
                print("Use GPU mode.")
            else:
                print("CUDA is not available on your machine. Use CPU mode instead.")

        if algo not in {'mu', 'hals', 'halsvar', 'bpp'}:
            raise ValueError("Parameter algo must be a valid value from ['mu', 'hals', 'halsvar', 'bpp']!")
        if mode not in {'batch', 'online'}:
            raise ValueError("Parameter mode must be a valid value from ['batch', 'online']!")
        if beta_loss != 2 and mode == 'online':
            print("Cannot perform online update when beta not equal to 2. Switch to batch update method.")
            mode = 'batch'

        if algo == 'hals':
            batch_hals_max_iter = 1

        model_class = None
        kwargs = {'alpha_W': alpha_W, 'l1_ratio_W': l1_ratio_W, 'alpha_H': alpha_H, 'l1_ratio_H': l1_ratio_H,
                  'fp_precision': fp_precision, 'device_type': device_type}

        if mode == 'batch':
            kwargs['max_iter'] = batch_max_iter
            if algo == 'mu':
                model_class = NMFBatchMU
            elif algo == 'hals' or algo == 'halsvar':
                model_class = NMFBatchHALS
                kwargs['hals_tol'] = batch_hals_tol
                kwargs['hals_max_iter'] = batch_hals_max_iter
            else:
                model_class = NMFBatchNnlsBpp
        else:
            kwargs['max_pass'] = online_max_pass
            kwargs['chunk_size'] = online_chunk_size
            if algo == 'mu' or algo == 'hals' or algo == 'halsvar':
                kwargs['chunk_max_iter'] = online_chunk_max_iter
                kwargs['h_tol'] = online_h_tol
                kwargs['w_tol'] = online_w_tol
                model_class = NMFOnlineMU if algo == 'mu' else NMFOnlineHALS
            else:
                model_class = NMFOnlineNnlsBpp

        model = model_class(
            n_components=n_components,
            init=init,
            beta_loss=beta_loss,
            tol=tol,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs
        )
        self.model = model

    def fit(self, X, verbose=False):
        self.model.fit(X, verbose)

    def fit_transform(self, X):
        return self.model.fit_transform(X, verbose=False)

    @staticmethod
    def transform(X, W, device='cpu'):
        '''

        :param X: N x D
        :param W: K x D
        :return: H: K x N
        '''

        CTC = W @ W.T
        CTB = W @ X.T
        H = torch.zeros((W.shape[0], X.shape[0]))
        n_iters = nnls_bpp(CTC, CTB, H, device)
        return H.T