import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from scipy.sparse import csc_matrix
from utils_dataset import PredictDataset
import scipy

def create_explain_matrix(input_dim, cat_emb_dim, cat_idxs, post_embed_dim):
    """
    This is a computational trick.
    In order to rapidly sum importances from same embeddings
    to the initial index.
    Parameters
    ----------
    input_dim : int
        Initial input dim
    cat_emb_dim : int or list of int
        if int : size of embedding for all categorical feature
        if list of int : size of embedding for each categorical feature
    cat_idxs : list of int
        Initial position of categorical features
    post_embed_dim : int
        Post embedding inputs dimension
    Returns
    -------
    reducing_matrix : np.array
        Matrix of dim (post_embed_dim, input_dim)  to performe reduce
    """

    if isinstance(cat_emb_dim, int):
        all_emb_impact = [cat_emb_dim - 1] * len(cat_idxs)
    else:
        all_emb_impact = [emb_dim - 1 for emb_dim in cat_emb_dim]

    acc_emb = 0
    nb_emb = 0
    indices_trick = []
    for i in range(input_dim):
        if i not in cat_idxs:
            indices_trick.append([i + acc_emb])
        else:
            indices_trick.append(
                range(i + acc_emb, i + acc_emb + all_emb_impact[nb_emb] + 1)
            )
            acc_emb += all_emb_impact[nb_emb]
            nb_emb += 1

    reducing_matrix = np.zeros((post_embed_dim, input_dim))
    for i, cols in enumerate(indices_trick):
        reducing_matrix[cols, i] = 1

    return scipy.sparse.csc_matrix(reducing_matrix)


class TabExpModel(object) :
    def __init__(self, network,batch_size=100,device="cpu") :
        self.network = network
        self.batch_size = batch_size
        self.reducing_matrix = create_explain_matrix(
            self.network.input_dim,
            self.network.cat_emb_dim,
            self.network.cat_idxs,
            self.network.post_embed_dim,
        )
        self.device = device 
        
    def explain(self, X):
        """
        Return local explanation
        Parameters
        ----------
        X : tensor: `torch.Tensor`
            Input data
        Returns
        -------
        M_explain : matrix
            Importance per sample, per columns.
        masks : matrix
            Sparse matrix showing attention masks used by network.
        """
        self.network.eval()

        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
        )

        res_explain = []

        for batch_nb, data in enumerate(dataloader):
            data = data.to(self.device).float()

            M_explain, masks = self.network.forward_masks(data)
            for key, value in masks.items():
                masks[key] = csc_matrix.dot(
                    value.cpu().detach().numpy(), self.reducing_matrix
                )

            res_explain.append(
                csc_matrix.dot(M_explain.cpu().detach().numpy(), self.reducing_matrix)
            )

            if batch_nb == 0:
                res_masks = masks
            else:
                for key, value in masks.items():
                    res_masks[key] = np.vstack([res_masks[key], value])

        res_explain = np.vstack(res_explain)

        return res_explain, res_masks
    
    def compute_feature_importances(self, X):
        """Compute global feature importance.
        Parameters
        ----------
        loader : `torch.utils.data.Dataloader`
            Pytorch dataloader.
        """
        dataloader = DataLoader(
            PredictDataset(X),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.network.eval()
        feature_importances_ = np.zeros((self.network.post_embed_dim))
        for data in dataloader:
            data = data.to(self.device).float()
            M_explain, masks = self.network.forward_masks(data)
            feature_importances_ += M_explain.sum(dim=0).cpu().detach().numpy()

        feature_importances_ = csc_matrix.dot(
            feature_importances_, self.reducing_matrix
        )
        feature_importances_ = feature_importances_ / np.sum(feature_importances_)
        return feature_importances_
    
    def show_importance_plot(self, X , columns = None, fig_kwargs={"figsize":(7,15)}) :
        fe_imp = self.compute_feature_importances(X)
        if columns is None :
            pd.DataFrame([fe_imp]).T.sort_values(by=0).plot.barh(figsize=(7,15))
        else:
            pd.DataFrame([fe_imp], columns = columns).T.sort_values(by=0).plot.barh(**fig_kwargs)