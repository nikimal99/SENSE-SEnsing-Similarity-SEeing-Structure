import os
os.environ["KEOPS_USE_GPU"] = "0"  # disable GPU for KeOps before importing anything else
import os
os.environ["LIBRARY_PATH"] = "/usr/local/cuda-12.1/lib64/stubs:" + os.environ.get("LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.1/lib64/stubs:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["CPLUS_INCLUDE_PATH"] = "/usr/local/cuda-12.1/include:" + os.environ.get("CPLUS_INCLUDE_PATH", "")
import pykeops
import sys
import numpy as np
import torch
import warnings
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision.models import resnet34, ResNet34_Weights
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_digits
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from scipy.linalg import fractional_matrix_power
from scipy.spatial.distance import cdist, pdist
import seaborn as sns
import phate
import scprep
from tqdm import tqdm
import argparse
import datetime
import time
from sklearn.neighbors import NearestNeighbors
from rnaseq_datasets import get_rnaseq_dataset
from create_dataset_sense import plot_samples, load_data
from numba.core.errors import NumbaWarning
warnings.filterwarnings("ignore", category=NumbaWarning)
sys.path.append(os.path.join(os.path.dirname(__file__), "contrastive-ne-master", "src", "cne"))
import main_cne
sys.path.append('/umap')
import umap.plot
from umap.utils import submatrix, average_nn_distance
from snc.snc import SNC
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# os.makedirs("logs", exist_ok=True)

def euclidean_dist_tensor(x, y=None):
    if y is None:
        y = x
    x_norm = (x**2).sum(1).view(-1, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    
    dist = torch.clamp(dist, min=0.0)
    return torch.sqrt(dist)

def Hbeta(D=np.array([]), beta=1.0):
    P = np.exp(-D.copy() * beta)
    # sumP = sum(P)
    sumP = np.sum(P)
    if sumP == 0:
        sumP = 1e-10  # or some small epsilon
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    # H = np.log(sumP) + beta * np.sum(D * P) / sumP
    # P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    # print("distamce matrix", D)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    print("P complete", P, P.shape)
    return P

def tsne(X=np.array([]), no_dims=2, perplexity=30.0, random_state=None,max_iter=1000, initial_momentum=0.4, final_momentum=0.8, eta=100, min_gain=0.01, early_exag=2):

    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    (n_tsne, d) = X.shape
    Y = np.random.randn(n_tsne, no_dims)
    dY = np.zeros((n_tsne, no_dims))
    iY = np.zeros((n_tsne, no_dims))
    gains = np.ones((n_tsne, no_dims))

    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * early_exag							
    P = np.maximum(P, 1e-12)
    

    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n_tsne):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n_tsne, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / early_exag

    return Y


def dist_approx(*client_data, X_a, glo):
    num_clients = len(client_data)
    num_global_anchors = glo
    X_a_global = X_a[:num_global_anchors] 
    X_a_local = X_a[num_global_anchors:]  

    n_clients = [data.shape[0] for data in client_data]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_a_global_tensor = torch.tensor(X_a_global, dtype=torch.float32).to(device)
    X_a_local_tensors = [torch.tensor(X_a_local_clients[i], dtype=torch.float32).to(device) for i in range(num_clients)]
    client_data_tensors = [torch.tensor(data, dtype=torch.float32).to(device) for data in client_data]

    all_anchors = torch.cat([X_a_global_tensor, torch.cat(X_a_local_tensors)], dim=0)
    DA_anchors = euclidean_dist_tensor(all_anchors).cpu().numpy()

    D_size = sum(n_clients) + X_a_global.shape[0] + sum(x.shape[0] for x in X_a_local_clients.values())
    D_tensor = torch.zeros((D_size, D_size), device=device)

    global_anchor_start = sum(n_clients)
    local_anchor_start = global_anchor_start + X_a_global.shape[0]

    # Non-anchor ↔ Non-anchor (intra-client)
    start_idx = 0
    for i, C_na_tensor in enumerate(client_data_tensors):
        end_idx = start_idx + n_clients[i]
        DA_na = euclidean_dist_tensor(C_na_tensor)
        D_tensor[start_idx:end_idx, start_idx:end_idx] = DA_na  # Symmetric already

        # Inter-client non-anchor ↔ non-anchor = 0
        for j in range(i + 1, num_clients):
            j_start = sum(n_clients[:j])
            j_end = j_start + n_clients[j]
            D_tensor[start_idx:end_idx, j_start:j_end] = 0
            D_tensor[j_start:j_end, start_idx:end_idx] = 0

        start_idx = end_idx

    # Non-anchor ↔ Anchors
    start_idx = 0
    for i, C_na_tensor in enumerate(client_data_tensors):
        end_idx = start_idx + n_clients[i]

        # Global
        dna_global = euclidean_dist_tensor(C_na_tensor, X_a_global_tensor)
        D_tensor[start_idx:end_idx, global_anchor_start:local_anchor_start] = dna_global
        D_tensor[global_anchor_start:local_anchor_start, start_idx:end_idx] = dna_global.t()  # Transpose

        # Local
        dna_local = euclidean_dist_tensor(C_na_tensor, torch.cat(X_a_local_tensors, dim=0))
        D_tensor[start_idx:end_idx, local_anchor_start:] = dna_local
        D_tensor[local_anchor_start:, start_idx:end_idx] = dna_local.t()

        # Mask other clients’ local anchors
        for j in range(num_clients):
            if i != j:
                local_start_j = local_anchor_start + sum(X_a_local_clients[k].shape[0] for k in range(j))
                local_end_j = local_start_j + X_a_local_clients[j].shape[0]
                D_tensor[start_idx:end_idx, local_start_j:local_end_j] = 0
                D_tensor[local_start_j:local_end_j, start_idx:end_idx] = 0

        start_idx = end_idx

    # Anchors ↔ Anchors (global, local, cross)
    D_tensor[global_anchor_start:local_anchor_start, global_anchor_start:local_anchor_start] = \
        torch.tensor(DA_anchors[:num_global_anchors, :num_global_anchors], device=device)
    
    D_tensor[local_anchor_start:, local_anchor_start:] = \
        torch.tensor(DA_anchors[num_global_anchors:, num_global_anchors:], device=device)

    D_tensor[global_anchor_start:local_anchor_start, local_anchor_start:] = \
        torch.tensor(DA_anchors[:num_global_anchors, num_global_anchors:], device=device)
    
    D_tensor[local_anchor_start:, global_anchor_start:local_anchor_start] = \
        torch.tensor(DA_anchors[num_global_anchors:, :num_global_anchors], device=device)  # Transpose explicitly

    # Final symmetrization (safety)
    D_tensor = (D_tensor + D_tensor.t()) / 2
    D = D_tensor.cpu().numpy()

    # === W_1 Matrix ===
    W_1_tensor = torch.zeros((D_size, D_size), device=device)
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + n_clients[i]

        # Global anchors
        W_1_tensor[start_idx:end_idx, global_anchor_start:local_anchor_start] = 1
        W_1_tensor[global_anchor_start:local_anchor_start, start_idx:end_idx] = 1

        # Local anchors
        local_start_i = local_anchor_start + sum(X_a_local_clients[j].shape[0] for j in range(i))
        local_end_i = local_start_i + X_a_local_clients[i].shape[0]
        W_1_tensor[start_idx:end_idx, local_start_i:local_end_i] = 1
        W_1_tensor[local_start_i:local_end_i, start_idx:end_idx] = 1

        start_idx = end_idx

    # Anchors ↔ Anchors
    W_1_tensor[global_anchor_start:, global_anchor_start:] = 1

    W_1 = W_1_tensor.cpu().numpy()

    # === Laplacian V ===
    ones_vector = torch.ones(W_1_tensor.shape[0], device=device)
    diag_values = torch.matmul(W_1_tensor, ones_vector)
    diag_matrix = torch.diag(diag_values.cpu())
    V_tensor = diag_matrix - W_1_tensor.cpu()
    V_tensor = V_tensor.to(W_1_tensor.device)

    V = V_tensor.cpu().numpy()
    V1 = V[:sum(n_clients), :sum(n_clients)]
    V2 = V[:sum(n_clients), sum(n_clients):]

    return D, W_1, DA_anchors, V, V1, V2



def mat(D, V1, V2, W_1, DA, X_a, X_a_local_clients, glo, n_list, n, d, Zu_samples):

    X_a_global = X_a[:glo]
    X_a_local = X_a[glo:]

    X_a_global_tensor = torch.tensor(X_a_global, dtype=torch.float32).to(device)
    X_a_local_tensors = [torch.tensor(X_a_local_clients[i], dtype=torch.float32).to(device) for i in range(len(n_list))]

    Zu = []
    start_idx = 0
    for n_i in n_list:
        Zu_i = Zu_samples[start_idx:start_idx + n_i, :]
        Zu.append(Zu_i)
        start_idx += n_i
    Zu_tensors = [torch.tensor(z, dtype=torch.float32).to(device) for z in Zu]

    # Initialize final distance matrix
    total_size = sum(n_list) + DA.shape[0]
    D_new_tensor = torch.zeros((total_size, total_size), device=device)

    # Compute intra-client distances
    Dnew_list = [euclidean_dist_tensor(z_tensor).cpu().numpy() for z_tensor in Zu_tensors]
    DNA_new_global = [euclidean_dist_tensor(Zu_tensors[i], X_a_global_tensor).cpu().numpy() for i in range(len(n_list))]
    DNA_new_local = [euclidean_dist_tensor(Zu_tensors[i], X_a_local_tensors[i]).cpu().numpy() for i in range(len(n_list))]

    start_i = 0
    local_anchor_start = sum(n_list) + glo

    for i in range(len(n_list)):
        end_i = start_i + n_list[i]

        # Fill: Zu_i ↔ Zu_i
        D_new_tensor[start_i:end_i, start_i:end_i] = torch.tensor(Dnew_list[i], device=device)

        # # Explicitly zero: Zu_i ↔ Zu_j, j ≠ i
        for j in range(len(n_list)):
            if i != j:
                start_j = sum(n_list[:j])
                end_j = start_j + n_list[j]
                D_new_tensor[start_i:end_i, start_j:end_j] = 0
                D_new_tensor[start_j:end_j, start_i:end_i] = 0

        # Fill: Zu_i ↔ Global anchors
        D_new_tensor[start_i:end_i, sum(n_list):local_anchor_start] = torch.tensor(DNA_new_global[i], device=device)
        D_new_tensor[sum(n_list):local_anchor_start, start_i:end_i] = torch.tensor(DNA_new_global[i], device=device).T

        # Fill: Zu_i ↔ own local anchors
        local_start_i = local_anchor_start + sum(X_a_local_clients[j].shape[0] for j in range(i))
        local_end_i = local_start_i + X_a_local_clients[i].shape[0]
        D_new_tensor[start_i:end_i, local_start_i:local_end_i] = torch.tensor(DNA_new_local[i], device=device)
        D_new_tensor[local_start_i:local_end_i, start_i:end_i] = torch.tensor(DNA_new_local[i], device=device).T

        # Explicitly zero: Zu_i ↔ local anchors of other clients
        for j in range(len(n_list)):
            if i != j:
                local_start_j = local_anchor_start + sum(X_a_local_clients[k].shape[0] for k in range(j))
                local_end_j = local_start_j + X_a_local_clients[j].shape[0]
                D_new_tensor[start_i:end_i, local_start_j:local_end_j] = 0
                D_new_tensor[local_start_j:local_end_j, start_i:end_i] = 0

        start_i = end_i
    D_new_tensor[sum(n_list):, sum(n_list):] = torch.tensor(DA, device=device)

    return D_new_tensor

def anchor_connection_stats(W_1, n_clients, num_global_anchors, X_a_local_clients):


    total_na = sum(n_clients)
    total_anchors = num_global_anchors + sum([X_a_local_clients[i].shape[0] for i in range(len(n_clients))])

    global_anchor_start = total_na
    local_anchor_start = global_anchor_start + num_global_anchors

    log_message("\n--- Anchor Connectivity Stats per Client ---\n", log_file)

    start_idx = 0
    for i, n_na in enumerate(n_clients):
        end_idx = start_idx + n_na

        # Non-anchor block (client i)
        W_block = W_1[start_idx:end_idx, global_anchor_start:]  # connections to all anchors
        anchor_counts = np.sum(W_block > 0, axis=1)  # per non-anchor

        # Summarize
        avg_connected_anchors = np.mean(anchor_counts)
        min_connected = np.min(anchor_counts)
        max_connected = np.max(anchor_counts)

        log_message(f"Client {i+1}:", log_file)
        log_message(f"  Non-anchors: {n_na}", log_file)
        # print(f"  Avg connected anchors per non-anchor: {avg_connected_anchors:.2f}")
        log_message(f"  Min/Max connected: {min_connected} / {max_connected}", log_file)
        print()

        start_idx = end_idx

def maskedX_a(BZ2, V2_tensor, X_a_tensor, X_a_local_clients, num_global_anchors):
    """
    Construct a masked anchor matrix X_a_masked where:
    - global anchors are visible to all clients,
    - local anchors are only visible to their respective clients.

    BZ2 and V2_tensor are of shape (n_total_NA, total_anchors)
    X_a_tensor is of shape (total_anchors, d)
    """
    X_a_masked = torch.zeros_like(X_a_tensor)
    num_clients = len(X_a_local_clients)

    # Compute base offset
    local_anchor_start = num_global_anchors

    for i in range(num_clients):
        # Determine range of client i's local anchors inside X_a_tensor
        local_anchor_offset = sum(X_a_local_clients[j].shape[0] for j in range(i))
        local_start_i = local_anchor_start + local_anchor_offset
        local_end_i = local_start_i + X_a_local_clients[i].shape[0]

        # Copy global anchors (only once)
        if i == 0:
            X_a_masked[:num_global_anchors, :] = X_a_tensor[:num_global_anchors, :]

        # Copy only client i's own local anchors
        X_a_masked[local_start_i:local_end_i, :] = X_a_tensor[local_start_i:local_end_i, :]

    # Now multiply with structured Laplacian term
    term2 = torch.matmul(BZ2 - V2_tensor, X_a_masked)
    return term2




def MDS_X(D, V1, V2, W_1, DA, X_a,X_a_local_clients, num_global_anchors, n_list, n, d):
    # print("enter here in MDS chk D:", D.shape)
    D_inv = np.reciprocal(D, out=np.zeros_like(D), where=(D != 0))
    D_inv_tensor = torch.tensor(D_inv, dtype=torch.float32).to(device)
    np.random.seed(50)
    torch.manual_seed(50)
    Zu_samples = np.random.uniform(0, 1, size=(d, n + X_a.shape[0])).T
    # print("enter here in MDS chk Zu_samples and n_list:", Zu_samples.shape, n_list, len(n_list))
    
    V1_tensor = torch.tensor(V1, dtype=torch.float32).to(device)
    V2_tensor = torch.tensor(V2, dtype=torch.float32).to(device)

    epsilon = 1e-3
    epochs = 2000
    loss = []

    Zu_combined = np.vstack([Zu_samples[sum(n_list[:i]):sum(n_list[:i+1]), :] for i in range(len(n_list))])
    Zu_combined_tensor = torch.tensor(Zu_combined, dtype=torch.float32).to(device)
    D_new_tensor = mat(D,V1,V2,W_1,DA,X_a=X_a,X_a_local_clients=X_a_local_clients,glo=num_global_anchors,n_list=n_sizes,n=n,d=d,Zu_samples=Zu_samples)

    n_total = sum(n_list)
    n_anchors = X_a.shape[0]
    # print("enter here in MDS chk n_total, n_anchors:", n_total, n_anchors)
    X_a_tensor = torch.tensor(X_a, dtype=torch.float32).to(device)
    # print("chck in mDS X_a_tensor", X_a_tensor.shape)
    D_tensor = torch.tensor(D, dtype=torch.float32).to(device)
    W_1_tensor = torch.tensor(W_1, dtype=torch.float32).to(device)
    W_new_tensor = W_1_tensor * D_tensor
    
    W_final_tensor = torch.zeros_like(W_new_tensor, device=device)
    triu_indices = torch.triu_indices(D_inv_tensor.shape[0], D_inv_tensor.shape[1], offset=1)
    D_inv_new = torch.zeros_like(D_new_tensor, device=device)
    
    for t in tqdm(range(epochs)):

        W_final_tensor.zero_()
        nonzero_mask = D_new_tensor > 0
        
        W_final_tensor[nonzero_mask] = W_new_tensor[nonzero_mask] / D_new_tensor[nonzero_mask]
        row_sums = torch.sum(W_final_tensor, dim=1)

        B_Z_tensor = torch.diag(row_sums) - W_final_tensor
        # print("chck in MDS:B_Z_tensor, W_final_tensor ", B_Z_tensor.shape, W_final_tensor.shape)
        BZ1 = B_Z_tensor[:n_total, :n_total]
        BZ2 = B_Z_tensor[:n_total, n_total:n_total + n_anchors]
        # print("chck in MDS:BZ1, BZ2 ", BZ1.shape, BZ2.shape)
        # print("X_a_tensor",X_a_tensor)
        term1 = torch.matmul(BZ1, Zu_combined_tensor)
        term2 = maskedX_a(BZ2, V2_tensor, X_a_tensor, X_a_local_clients, num_global_anchors)
        # term2 = torch.matmul(BZ2 - V2_tensor, X_a_tensor)
        # print("chck in MDS:term1, term2 ", term1.shape, term2.shape)

        X_final_tensor = torch.linalg.solve(V1_tensor, term1 + term2)
        X_final = X_final_tensor.cpu().numpy()
        D_new_tensor = mat(D,V1,V2,W_1,DA,X_a=X_a,X_a_local_clients=X_a_local_clients,glo=num_global_anchors,n_list=n_sizes,n=n,d=d,Zu_samples=X_final)
        #D_inv_new = torch.zeros_like(D_new_tensor, device=device)
        D_inv_new.zero_()
        nonzero_mask = D_new_tensor != 0
        D_inv_new[nonzero_mask] = 1.0 / D_new_tensor[nonzero_mask]
        C_tensor = torch.square(D_tensor - D_new_tensor)
        W_upper = D_inv_new[triu_indices[0], triu_indices[1]]
        C_upper = C_tensor[triu_indices[0], triu_indices[1]]
        stress = torch.sum(W_upper * C_upper).item()
        loss.append(stress)
        Zu_combined_tensor = X_final_tensor
        # print(f"[t={t}] Stress: {stress:.4f}")
        # print("Max W_final_tensor:", W_final_tensor.max().item())
        # print("Min D_new_tensor (non-zero):", D_new_tensor[D_new_tensor > 0].min().item())
        # print("NaNs in X_final_tensor:", torch.isnan(X_final_tensor).any().item())
        if t % 10 == 0:
            print(f"Iteration {t}, Stress: {stress}")
        
        if t != 0 and abs(loss[t] - loss[t-1]) < epsilon:
            print(f"Converged at iteration {t}")
            break
    
    return X_final, loss

def dist_error(X_na, X_final):
    X_na_tensor = torch.tensor(X_na, dtype=torch.float32).to(device)
    X_final_tensor = torch.tensor(X_final, dtype=torch.float32).to(device)

    D_true = euclidean_dist_tensor(X_na_tensor).cpu().numpy()
    # z_true = spatial.distance.squareform(D_true)

    D_esti = euclidean_dist_tensor(X_final_tensor).cpu().numpy()
    # z_esti = spatial.distance.squareform(D_esti) 
    Error = np.linalg.norm((D_true - D_esti), 'fro')/ np.linalg.norm((D_true), 'fro')

    return Error, D_true, D_esti


def check_score(D_true, D_approx,k):
      f_scores = []
      for i in range(D_true.shape[0]):
        list1 =  np.argsort(D_true[i])
        list2 =  np.argsort(D_approx[i])
        newlist1 = list1[1:k]
        newlist2 = list2[1:k]
        count = 0
        for p in range(k-1):
          for q in range(k-1):
            if(newlist1[p] == newlist2[q]):
              count += 1
              break;
        
        f_score = 2*count/(2*count + (k- 1 - count))
        f_scores.append(f_score)
      avg_f_score = sum(f_scores)/len(f_scores)

      return avg_f_score

def measure(orig, emb, k=20, knn_ranking_info=None, return_local=False):
	if knn_ranking_info is None:
		orig_knn_indices, orig_ranking = knn_with_ranking(orig, k)
		emb_knn_indices,  emb_ranking  = knn_with_ranking(emb, k)
	else:
		orig_knn_indices, orig_ranking, emb_knn_indices, emb_ranking = knn_ranking_info

	if return_local:
		trust, local_trust = tnc_computation(orig_knn_indices, orig_ranking, emb_knn_indices, k, return_local)
		cont , local_cont  = tnc_computation(emb_knn_indices,  emb_ranking, orig_knn_indices, k, return_local)
		return ({
			"trustworthiness": trust,
			"continuity": cont
		}, {
			"local_trustworthiness": local_trust,
			"local_continuity": local_cont
		})
	else:
		trust = tnc_computation(orig_knn_indices, orig_ranking, emb_knn_indices, k, return_local)
		cont  = tnc_computation(emb_knn_indices,  emb_ranking, orig_knn_indices, k, return_local)
		return {
			"trustworthiness": trust,
			"continuity": cont
		}
    
def pairwise_distance_matrix(point, distance_function="euclidean"):
	if callable(distance_function):
		distance_matrix = cdist(point, point, distance_function)
	elif distance_function == "snn":
		## TODO
		pass
	else:
		distance_matrix = cdist(point, point, distance_function)
	return distance_matrix

def tnc_computation(base_knn_indices, base_ranking, target_knn_indices, k, return_local=False):
	local_distortion_list = []
	points_num = base_knn_indices.shape[0]

	for i in range(points_num):
		missings = np.setdiff1d(target_knn_indices[i], base_knn_indices[i])
		local_distortion = 0.0 
		for missing in missings:
			local_distortion += base_ranking[i, missing] - k
		local_distortion_list.append(local_distortion)
	local_distortion_list = np.array(local_distortion_list)
	local_distortion_list = 1 - local_distortion_list * (2 / (k * (2 * points_num - 3 * k - 1)))

	average_distortion = np.mean(local_distortion_list)

	if return_local:
		return average_distortion, local_distortion_list
	else:
		return average_distortion

def knn_with_ranking(points, k, distance_matrix=None):
  if distance_matrix is None:
    distance_matrix = pairwise_distance_matrix(points, "euclidean")

  knn_indices = np.empty((points.shape[0], k), dtype=np.int32)
  ranking = np.empty((points.shape[0], points.shape[0]), dtype=np.int32)
  
  for i in range(points.shape[0]):
    distance_to_i = distance_matrix[i]
    sorted_indices = np.argsort(distance_to_i)
    knn_indices[i] = sorted_indices[1:k+1]
    ranking[i] = np.argsort(sorted_indices)
  
  return knn_indices, ranking     

def log_message(message, log_file):
    with open(log_file, 'a') as f:
        f.write(str(message) + '\n')
    print(message)    





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run t-SNE with random search hyperparameters.')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations.')
    parser.add_argument('--initial_momentum', type=float, default=0.4, help='Initial momentum.')
    parser.add_argument('--final_momentum', type=float, default=0.8, help='Final momentum.')
    parser.add_argument('--eta', type=float, default=100, help='Learning rate (eta).')
    parser.add_argument('--min_gain', type=float, default=0.01, help='Minimum gain.')
    parser.add_argument('--early_exag', type=int, default=2, help='Early exaggeration.')
    parser.add_argument('--output', type=str, help='Output filename for the visualization')
    parser.add_argument('--dataset_name', type=str, choices=['BRCA', 'MNIST', 'rnaseq', 'german_credit', 'cifar10', 'fashionmnist', 'xin', 'baron_mouse', 'baron_human', 'muraro', 
                                                            'segerstolpe', 'amb', 'tm', 'zheng', 'DermaMNIST', 'PneumoniaMNIST', 'RetinaMNIST', 'BreastMNIST', 'BloodMNIST','OrganCMNIST', 'OrganSMNIST', 'OrganMNIST3D', 'FractureMNIST3D', 'breast_cancer_uci','RDkit', 'cora', 'citeseer', 'pubmed', 'uci_data_taiwanese', 'Gowalla', 'Foursquare'], required=True, 
                                                            help='Name of the dataset to load.')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients to use for the dataset.')
    parser.add_argument('--n_samples_per_node', type=int, default=None, help='Number of samples on each client.')
    parser.add_argument('--iid', action='store_true', help='Set to enable IID; omit for non-IID setting.')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
    parser.add_argument('--data_directory', type=str, default='../Dataset', help='Directory where dataset files are located.')
    parser.add_argument('--num_data', type=int, default=15000, help='subdata to use from full dataset')
    parser.add_argument('--colormap', type=str, default=None, help='Optional colormap override')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test split ratio.')
    parser.add_argument('--balanced', action='store_true', help='Set for balanced non-IID split (only used if not IID).')
    parser.add_argument('--seed', type=int, default=42, help='random seed .')

    args = parser.parse_args()
    max_iter = args.max_iter
    initial_momentum = args.initial_momentum
    final_momentum = args.final_momentum
    eta = args.eta
    min_gain = args.min_gain
    early_exag = args.early_exag
    dataset_name = args.dataset_name
    data_directory = args.data_directory
    num_clients = args.num_clients
    n_samples_per_node = args.n_samples_per_node
    iid = args.iid
    balanced= args.balanced
    alpha = args.alpha
    num_data =args.num_data
    seed=args.seed
    test_size = args.test_size

    colormap_dict = {
        'DermaMNIST': 'tab20',
        'PneumoniaMNIST': 'Set2',
        'RetinaMNIST': 'Dark2',
        'BreastMNIST': 'viridis',
        'BloodMNIST': 'plasma',
        'OrganCMNIST': 'Spectral',
        'OrganSMNIST': 'Accent',
        'OrganMNIST3D': 'cividis',
        'FractureMNIST3D': 'nipy_spectral',
        'german_credit': 'turbo',
        'xin': 'spring',
        'baron_mouse': 'winter',
        'baron_human': 'summer',
        'muraro': 'coolwarm',
        'segerstolpe': 'Wistia',
        'amb': 'hot',
        'tm': 'gist_earth',
        'zheng': 'brg'
    }

    # Attach the colormap to args for use later
    if args.colormap is None:
        args.colormap = colormap_dict.get(args.dataset_name, 'tab20')

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


    if args.iid:
        mode_tag = "iid"
    elif hasattr(args, "balanced") and args.balanced:
        mode_tag = "balanced_noniid"
    else:
        mode_tag = "unbalanced_noniid"

    # Base results directory
    base_result_dir = "/results"

    # Optional: Include test size in log path only for CBME datasets
    cbme_datasets = [
        "xin", "baron_mouse", "baron_human", "muraro",
        "segerstolpe", "amb", "tm", "zheng"
    ]

    if args.dataset_name in cbme_datasets and hasattr(args, "test_size"):
        test_frac = int(float(args.test_size) * 100)
        dataset_dir = os.path.join(base_result_dir, "cbme_varying_testsize", args.dataset_name, f"test{test_frac}")
    else:
        dataset_dir = os.path.join(base_result_dir, args.dataset_name)

    # Ensure directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    # Compose full log file path
    log_file = os.path.join(
        dataset_dir,
        f"results_log_{args.dataset_name}_{mode_tag}_{timestamp}.txt"
    )

    if args.output is None:
        args.output = f".../{dataset_name}_anchorvis.png"
    
    print("dataset_name", dataset_name)
    train_data, test_data = load_data(args, num_data=num_data, test_size=test_size, dataset_name=dataset_name, num_clients=num_clients, data_dir=data_directory)

    log_message(f"train_data: {len(train_data)}", log_file)
    log_message(f"test_data: {len(test_data)}", log_file)

    train_features = []
    train_labels = []
    test_features = []
    test_labels = []
    

    for client_id in range(len(train_data)):
        train_features.append(train_data[client_id][0])
        train_labels.append(train_data[client_id][1])
        test_features.append(test_data[client_id][0])
        test_labels.append(test_data[client_id][1])



    for client_id in range(len(train_data)):
        globals()[f'C{client_id + 1}_na_data'] = train_features[client_id]  
        globals()[f'C{client_id + 1}_na_labels'] = train_labels[client_id]  
        globals()[f'C{client_id + 1}_a_data'] = test_features[client_id] 
        globals()[f'C{client_id + 1}_a_labels'] = test_labels[client_id]

    # Prepare the labels
    labels = np.concatenate([globals()[f'C{i + 1}_na_labels'] for i in range(len(train_data))])


    X_na = np.concatenate([globals()[f'C{i + 1}_na_data'] for i in range(len(train_data))], axis=0)
    all_test_features = np.concatenate([globals()[f'C{i + 1}_a_data'] for i in range(len(test_data))], axis=0)
    all_test_labels = np.concatenate([globals()[f'C{i + 1}_a_labels'] for i in range(len(test_data))], axis=0)

    total_test_samples = all_test_features.shape[0]

    dim = X_na.shape[1] 
    print("dimension of data", dim, X_na.shape, total_test_samples)  
    if dim > total_test_samples:
        num_anchors_per_client = total_test_samples//num_clients
    else:
        num_anchors_per_client = dim - 1  # Each client gets (dim - 1) anchors total
    print("num_anchors_per_client", num_anchors_per_client)     

    num_global_anchors = num_anchors_per_client // 2
    num_local_anchors_per_client = num_anchors_per_client - num_global_anchors

    log_message(f"Total anchors per client: {num_anchors_per_client} (global: {num_global_anchors}, local: {num_local_anchors_per_client})", log_file)

    permutation = np.random.permutation(total_test_samples)
    X_a_global = all_test_features[permutation[:num_global_anchors]]
    y_a_global = all_test_labels[permutation[:num_global_anchors]]

    remaining_indices = permutation[num_global_anchors:]
    remaining_test_features = all_test_features[remaining_indices]
    remaining_test_labels = all_test_labels[remaining_indices]

    X_a_local_clients = {}
    y_a_local_clients = {}

    for client_id in range(len(test_data)):
        start_idx = client_id * num_local_anchors_per_client
        end_idx = start_idx + num_local_anchors_per_client
        
        X_a_local_client = remaining_test_features[start_idx:end_idx]
        y_a_local_client = remaining_test_labels[start_idx:end_idx]
        
        X_a_local_clients[client_id] = X_a_local_client
        y_a_local_clients[client_id] = y_a_local_client

    # Now each client gets:
    # - X_a_global (same for all clients)
    # - X_a_local_clients[client_id] (specific for that client)

    # Example to access
    for client_id in range(len(test_data)):
        log_message(f"\nClient {client_id + 1}:", log_file)
        log_message(f"  Global anchors shape: {X_a_global.shape}", log_file)
        log_message(f"  Local anchors shape: {X_a_local_clients[client_id].shape}", log_file)
        total = X_a_global.shape[0] + X_a_local_clients[client_id].shape[0]
        log_message(f"  Total anchors received by client {client_id + 1}: {total}", log_file)
        assert total == num_anchors_per_client, f"Mismatch in total anchors for client {client_id+1}"

    # # print("shapes of X_na and X_a:", X_na.shape, X_a.shape)
    # log_message(f"shapes of X_na: {X_na.shape}", log_file)
    # log_message(f"shapes of X_a: {X_a.shape}", log_file)
    # X_combined = np.vstack((X_na, X_a))
    # # print("values of data", X_na, X_a)

    scaler = MinMaxScaler()
    scaler.fit(X_na)
    X_na = scaler.transform(X_na)
    X_a_global = scaler.transform(X_a_global)
    for client_id in range(len(test_data)):
        X_a_local_clients[client_id] = scaler.transform(X_a_local_clients[client_id])


    start_idx = 0
    for client_id in range(len(train_data)):
          end_idx = start_idx + train_features[client_id].shape[0]
          globals()[f'C{client_id + 1}_na_data'] = X_na[start_idx:end_idx]
          start_idx = end_idx 
     
    n = X_na.shape[0] 
    d = X_na.shape[1] 

    n_sizes = [train_features[i].shape[0] for i in range(len(train_features))]  
    for i in range(len(n_sizes)):
        globals()[f'n{i + 1}'] = n_sizes[i]  

    log_message(f"X_a_global shape: {X_a_global.shape}", log_file)
    for client_id, local_anchors in X_a_local_clients.items():
        log_message(f"Client {client_id} local anchors shape: {local_anchors.shape}", log_file)

    X_a = np.concatenate([X_a_global] + list(X_a_local_clients.values()), axis=0)  
    print("lets check X_a", X_a.shape)
    m = X_a.shape[0] 
    log_message(f"Number of anchors Samples (m): {m}",log_file)
    log_message(f"Number of non anchors Samples (n): {n}", log_file)
    log_message(f"Number of Features (d): {d}", log_file)

    log_message(f"Number of anchors Samples (m): {m}", log_file)
    log_message(f"Number of non anchors Samples (n): {n}", log_file)
    log_message(f"Number of Features (d): {d}", log_file)
    client_data = [globals()[f'C{i + 1}_na_data'].astype('float32') for i in range(len(train_data))] 
    
#     # print("checking all the data: X_na, X_a, d, train_data, test_Data", X_na.shape, X_a.shape, d, len(train_labels), len(test_labels))

#     if d < X_combined.shape[0]:
#         X_a = X_a[:d-1,:]
#     else:
#         X_a = X_a    

#     print("X_a into MDS", X_a.shape)   
    start_time = time.time()
    D, W_1, DA_anchors, V, V1, V2 = dist_approx(*client_data, X_a=X_a, glo=X_a_global.shape[0])
    anchor_connection_stats(W_1, n_sizes, num_global_anchors, X_a_local_clients)

    print("matrices: D, DA, V, V1, V2, W_1,", D.shape, DA_anchors.shape, V.shape, V1.shape, V2.shape, W_1.shape)
    
#     # Perform MDS
    X_final, loss = MDS_X(D, V1, V2, W_1, DA_anchors, X_a.astype('float32'), X_a_local_clients,num_global_anchors, n_sizes, n, d)

    np.savez("mds_embedding_output_mnist.npz", X_na=X_na, X_final=X_final, labels=labels)
    print("Saved X_na and X_final to mds_embedding_output.npz")

    end_time = time.time()
    log_message(f"Computation time: {end_time - start_time:.2f} seconds", log_file)
    
    print("Calculating the distance error between X_na and X_final")
    error, D_true, D_esti= dist_error(X_na, X_final)
    fscore = check_score(D_true, D_esti, 11)

#     assert not np.isnan(X_final).any(), "NaN in input features"
#     assert not np.isinf(X_final).any(), "Inf in input features"

    ##-------visualization using PHATE 
    cmap_used = args.colormap

    # ------- PHATE Visualization on Original Data -------
    phate_op = phate.PHATE(gamma=0.8, knn=7)
    data_phate_orig = phate_op.fit_transform(X_na)

    ax = scprep.plot.scatter2d(
        data_phate_orig,
        c=labels,
        figsize=(10, 8),
        cmap=cmap_used,
        ticks=False,
        s=12,
        alpha=0.85,
        label_prefix="PHATE"
    )
    fig = ax.figure
    phate_output_file = args.output.replace('.png', '_origphate.png')
    fig.savefig(phate_output_file, dpi=300, bbox_inches='tight')
    plt.show()
    log_message(f"orig data PHATE plot saved to: {phate_output_file}", log_file)

    # ------- t-SNE Visualization on Original Data -------
    Y_tsne_orig = tsne(X_na, 2, max_iter=max_iter, initial_momentum=initial_momentum,
                final_momentum=final_momentum, eta=eta, min_gain=min_gain, early_exag=early_exag)
    plt.figure(figsize=(10, 8))
    plt.scatter(Y_tsne_orig[:, 0], Y_tsne_orig[:, 1], c=labels, cmap=cmap_used, s=12, alpha=0.85, edgecolor='none')
    plt.title("orig data t-SNE Projection", fontsize=16, fontweight='bold')
    plt.axis('off')
    tsne_output_file = args.output.replace('.png', '_origtsne.png')
    plt.savefig(tsne_output_file, dpi=300, bbox_inches='tight')
    plt.show()
    log_message(f"orig data t-SNE plot saved to: {tsne_output_file}", log_file)

    # ------- UMAP Visualization on Original Data -------
    mapper = umap.UMAP(n_neighbors=10, min_dist=0.2, metric='correlation', init='spectral').fit(X_na)
    Y_umap_orig = mapper.embedding_
    plt.figure(figsize=(10, 8))
    plt.scatter(Y_umap_orig[:, 0], Y_umap_orig[:, 1], c=labels, cmap=cmap_used, s=12, alpha=0.85, edgecolor='none')
    plt.title("orig data UMAP Projection", fontsize=16, fontweight='bold')
    plt.axis('off')
    umap_output_file = args.output.replace('.png', '_origumap.png')
    plt.savefig(umap_output_file, dpi=300, bbox_inches='tight')
    plt.show()
    log_message(f"orig data UMAP plot saved to: {umap_output_file}", log_file)

    # ------- CNE Spectrum Visualization on Original Data -------
    spec_params = [0.0, 0.5, 1.0]
    neg_embeddings_orig = {}

    for s in spec_params:
        embedder = main_cne.CNE(loss_mode="neg", s=s)
        embd_orig = embedder.fit_transform(X_na)
        neg_embeddings_orig[s] = embd_orig

    fig, ax = plt.subplots(1, len(spec_params), figsize=(6 * len(spec_params), 5), constrained_layout=True)
    for i, s in enumerate(spec_params):
        ax[i].scatter(*neg_embeddings_orig[s].T, c=labels, cmap=cmap_used, alpha=0.85, s=10, edgecolor='none')
        ax[i].set_aspect("equal", "datalim")
        ax[i].axis("off")
        ax[i].set_title(f"s = {s}", fontsize=14, fontweight='bold')

    fig.suptitle(f"orig data CNE Spectrum Projection for {dataset_name.upper()}", fontsize=16, fontweight='bold')
    cne_output_file = args.output.replace('.png', '_orig_cne_spectrum.png')
    plt.savefig(cne_output_file, dpi=300, bbox_inches="tight")
    plt.show()
    log_message(f"orig data CNE spectrum plot saved to: {cne_output_file}", log_file)


    ##-------visualization using PHATE 
    cmap_used = args.colormap

    # ------- PHATE Visualization -------
    phate_op = phate.PHATE(gamma=0.8, knn=7)
    data_phate = phate_op.fit_transform(X_final)

    ax = scprep.plot.scatter2d(
        data_phate,
        c=labels,
        figsize=(10, 8),
        cmap=cmap_used,
        ticks=False,
        s=12,
        alpha=0.85,
        label_prefix="PHATE"
    )
    fig = ax.figure
    phate_output_file = args.output.replace('.png', '_phate.png')
    fig.savefig(phate_output_file, dpi=300, bbox_inches='tight')
    plt.show()
    log_message(f"PHATE plot saved to: {phate_output_file}", log_file)

    # ------- t-SNE Visualization -------
    Y_tsne = tsne(X_final, 2, max_iter=max_iter, initial_momentum=initial_momentum,
                final_momentum=final_momentum, eta=eta, min_gain=min_gain, early_exag=early_exag)
    plt.figure(figsize=(10, 8))
    plt.scatter(Y_tsne[:, 0], Y_tsne[:, 1], c=labels, cmap=cmap_used, s=12, alpha=0.85, edgecolor='none')
    plt.title("t-SNE Projection", fontsize=16, fontweight='bold')
    plt.axis('off')
    tsne_output_file = args.output.replace('.png', '_tsne.png')
    plt.savefig(tsne_output_file, dpi=300, bbox_inches='tight')
    plt.show()
    log_message(f"t-SNE plot saved to: {tsne_output_file}", log_file)

    # ------- UMAP Visualization -------
    mapper = umap.UMAP(n_neighbors=10, min_dist=0.2, metric='correlation', init='spectral').fit(X_final)
    Y_umap = mapper.embedding_
    plt.figure(figsize=(10, 8))
    plt.scatter(Y_umap[:, 0], Y_umap[:, 1], c=labels, cmap=cmap_used, s=12, alpha=0.85, edgecolor='none')
    plt.title("UMAP Projection", fontsize=16, fontweight='bold')
    plt.axis('off')
    umap_output_file = args.output.replace('.png', '_umap.png')
    plt.savefig(umap_output_file, dpi=300, bbox_inches='tight')
    plt.show()
    log_message(f"UMAP plot saved to: {umap_output_file}", log_file)

    # ------- CNE Spectrum Visualization -------
    spec_params = [0.0, 0.5, 1.0]
    neg_embeddings = {}

    for s in spec_params:
        embedder = main_cne.CNE(loss_mode="neg", s=s)
        embd = embedder.fit_transform(X_final)
        neg_embeddings[s] = embd

    fig, ax = plt.subplots(1, len(spec_params), figsize=(6 * len(spec_params), 5), constrained_layout=True)
    for i, s in enumerate(spec_params):
        ax[i].scatter(*neg_embeddings[s].T, c=labels, cmap=cmap_used, alpha=0.85, s=10, edgecolor='none')
        ax[i].set_aspect("equal", "datalim")
        ax[i].axis("off")
        ax[i].set_title(f"s = {s}", fontsize=14, fontweight='bold')

    fig.suptitle(f"CNE Spectrum Projection for {dataset_name.upper()}", fontsize=16, fontweight='bold')
    cne_output_file = args.output.replace('.png', '_cne_spectrum.png')
    plt.savefig(cne_output_file, dpi=300, bbox_inches="tight")
    plt.show()
    log_message(f"CNE spectrum plot saved to: {cne_output_file}", log_file)


# #     # --- Print Final Scores ---
    log_message(f"Calculating the distance error between X_na and X_final of shapes {X_na.shape}, {X_final.shape}", log_file)
    log_message(f"Error in distance approximation: {error}", log_file)
    log_message(f"F-score: {fscore}", log_file)

    ####------- phate metric on original data --------------------------------------
    log_message("Calculating phate metricson original data ...", log_file)
    parameter = {"k": 'sqrt', "alpha": 0.1}
    metrics_phate = SNC(raw=X_na.astype('float'), emb=data_phate_orig, iteration=300, dist_parameter=parameter)
    metrics_phate.fit()

    trust_phate = trustworthiness(X_na, data_phate_orig, n_neighbors=7)
    metrics_cont_phate = measure(X_na, data_phate_orig, k=7)

    log_message(f"orig data phate SNC metrics - Steadiness: {metrics_phate.steadiness()}, Cohesiveness: {metrics_phate.cohesiveness()}", log_file)
    log_message(f"orig data phate Trustworthiness: {trust_phate}", log_file)
    log_message(f"orig data phate Trustworthiness repo : {metrics_cont_phate['trustworthiness']:.4f}", log_file)
    log_message(f"orig data phate Continuity: {metrics_cont_phate['continuity']:.4f}", log_file)


####------- phate metric on SENSE data --------------------------------------
    log_message("Calculating phate metrics on SENSE data ...", log_file)
    parameter = {"k": 'sqrt', "alpha": 0.1}
    metrics_phate = SNC(raw=X_na.astype('float'), emb=data_phate, iteration=300, dist_parameter=parameter)
    metrics_phate.fit()

    trust_phate = trustworthiness(X_na, data_phate, n_neighbors=7)
    metrics_cont_phate = measure(X_na, data_phate, k=7)

    log_message(f"phate SNC metrics - Steadiness: {metrics_phate.steadiness()}, Cohesiveness: {metrics_phate.cohesiveness()}", log_file)
    log_message(f"phate Trustworthiness: {trust_phate}", log_file)
    log_message(f"phate Trustworthiness repo : {metrics_cont_phate['trustworthiness']:.4f}", log_file)
    log_message(f"phate Continuity: {metrics_cont_phate['continuity']:.4f}", log_file)


    # --- t-SNE Metrics original data ---
    log_message("Calculating t-SNE metrics on original data...", log_file)
    parameter = {"k": 'sqrt', "alpha": 0.1}
    metrics_tsne = SNC(raw=X_na.astype('float'), emb=Y_tsne_orig, iteration=300, dist_parameter=parameter)
    metrics_tsne.fit()

    trust_tsne = trustworthiness(X_na, Y_tsne_orig, n_neighbors=7)
    metrics_cont_tsne = measure(X_na, Y_tsne_orig, k=7)

    log_message(f" orig data t-SNE SNC metrics - Steadiness: {metrics_tsne.steadiness()}, Cohesiveness: {metrics_tsne.cohesiveness()}", log_file)
    log_message(f"orig data t-SNE Trustworthiness: {trust_tsne}", log_file)
    log_message(f"orig data tsne Trustworthiness repo : {metrics_cont_tsne['trustworthiness']:.4f}", log_file)
    log_message(f" orig data t-SNE Continuity: {metrics_cont_tsne['continuity']:.4f}", log_file)


    # --- t-SNE Metrics SENSE data ---
    log_message("Calculating t-SNE metrics on SENSE data...", log_file)
    parameter = {"k": 'sqrt', "alpha": 0.1}
    metrics_tsne = SNC(raw=X_na.astype('float'), emb=Y_tsne, iteration=300, dist_parameter=parameter)
    metrics_tsne.fit()

    trust_tsne = trustworthiness(X_na, Y_tsne, n_neighbors=7)
    metrics_cont_tsne = measure(X_na, Y_tsne, k=7)

    log_message(f"t-SNE SNC metrics - Steadiness: {metrics_tsne.steadiness()}, Cohesiveness: {metrics_tsne.cohesiveness()}", log_file)
    log_message(f"t-SNE Trustworthiness: {trust_tsne}", log_file)
    log_message(f"tsne Trustworthiness repo : {metrics_cont_tsne['trustworthiness']:.4f}", log_file)
    log_message(f" t-SNE Continuity: {metrics_cont_tsne['continuity']:.4f}", log_file)


    # --- UMAP Metrics original data---
    log_message("Calculating UMAP metrics on original data ...", log_file)
    metrics_umap = SNC(raw=X_na.astype('float'), emb=Y_umap_orig, iteration=300, dist_parameter=parameter)
    metrics_umap.fit()

    trust_umap = trustworthiness(X_na, Y_umap_orig, n_neighbors=7)
    metrics_cont_umap = measure(X_na, Y_umap_orig, k=7)

    log_message(f"orig data UMAP SNC metrics - Steadiness: {metrics_umap.steadiness()}, Cohesiveness: {metrics_umap.cohesiveness()}", log_file)
    log_message(f"orig data UMAP Trustworthiness: {trust_umap}", log_file )
    log_message(f"orig data umap Trustworthiness repo : {metrics_cont_umap['trustworthiness']:.4f}", log_file)
    log_message(f" orig data UMAP Continuity: {metrics_cont_umap['continuity']:.4f}", log_file)


    # --- UMAP Metrics SENSE data---
    log_message("Calculating UMAP metrics on SENSE data...", log_file)
    metrics_umap = SNC(raw=X_na.astype('float'), emb=Y_umap, iteration=300, dist_parameter=parameter)
    metrics_umap.fit()

    trust_umap = trustworthiness(X_na, Y_umap, n_neighbors=7)
    metrics_cont_umap = measure(X_na, Y_umap, k=7)

    log_message(f"UMAP SNC metrics - Steadiness: {metrics_umap.steadiness()}, Cohesiveness: {metrics_umap.cohesiveness()}", log_file)
    log_message(f"UMAP Trustworthiness: {trust_umap}", log_file )
    log_message(f"umap Trustworthiness repo : {metrics_cont_umap['trustworthiness']:.4f}", log_file)
    log_message(f" UMAP Continuity: {metrics_cont_umap['continuity']:.4f}", log_file)

    # --- CNE Metrics ---

# --- CNE Metrics original data ---
    log_message("Calculating original data CNE metrics...", log_file)
    for s in spec_params:
        embd_orig = neg_embeddings_orig[s]
        metrics_cne = SNC(raw=X_na.astype('float'), emb=embd_orig, iteration=300, dist_parameter=parameter)
        metrics_cne.fit()

        trust_cne = trustworthiness(X_na, embd_orig, n_neighbors=7)
        metric_cont_cne = measure(X_na, embd_orig, k=7)

        log_message(f"orig data CNE (s={s}) SNC metrics - Steadiness: {metrics_cne.steadiness()}, Cohesiveness: {metrics_cne.cohesiveness()}", log_file)
        log_message(f"orig data CNE (s={s}) Trustworthiness: {trust_cne}", log_file)
        log_message(f"orig data CNE Trustworthiness repo : {metric_cont_cne['trustworthiness']:.4f}", log_file)
        log_message(f"orig data CNE (s={s}) Continuity: {metric_cont_cne['continuity']:.4f}", log_file)

# --- CNE Metrics SENSE data ---
    log_message("Calculating CNE metrics on SENSE data...", log_file)
    for s in spec_params:
        embd = neg_embeddings[s]
        metrics_cne = SNC(raw=X_na.astype('float'), emb=embd, iteration=300, dist_parameter=parameter)
        metrics_cne.fit()

        trust_cne = trustworthiness(X_na, embd, n_neighbors=7)
        metric_cont_cne= measure(X_na, embd, k=7)

        log_message(f"CNE (s={s}) SNC metrics - Steadiness: {metrics_cne.steadiness()}, Cohesiveness: {metrics_cne.cohesiveness()}", log_file)
        log_message(f"CNE (s={s}) Trustworthiness: {trust_cne}", log_file)
        log_message(f"CNE Trustworthiness repo : {metric_cont_cne['trustworthiness']:.4f}", log_file)
        log_message(f"cne Continuity: {metric_cont_cne['continuity']:.4f}", log_file)

