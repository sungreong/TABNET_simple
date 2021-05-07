from scipy.sparse import csc_matrix
import numpy as np
"""
network 나 device 정의 필요 
"""
def _compute_feature_importances(loader):
    """Compute global feature importance.

    Parameters
    ----------
    loader : `torch.utils.data.Dataloader`
        Pytorch dataloader.

    """
    network.eval()
    feature_importances_ = np.zeros((network.post_embed_dim))
    for data, targets in loader:
        data = data.to(device).float()
        M_explain, masks = network.forward_masks(data)
        feature_importances_ += M_explain.sum(dim=0).cpu().detach().numpy()

    feature_importances_ = csc_matrix.dot(
        feature_importances_, reducing_matrix
    )
    feature_importances_ = feature_importances_ / np.sum(feature_importances_)
    return feature_importances_


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

reducing_matrix = create_explain_matrix(
            network.input_dim,
            network.cat_emb_dim,
            network.cat_idxs,
            network.post_embed_dim,
        )
reducing_matrix
