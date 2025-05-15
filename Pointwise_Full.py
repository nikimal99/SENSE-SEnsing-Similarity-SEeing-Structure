import numpy as np
import pandas as pd
import random
# import tensorflow as tf
import warnings
from math import log2
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy import spatial
from scipy import sparse as sp
from tqdm.notebook import tqdm
import numpy as np
from torchvision import datasets, transforms
import torch
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
from scipy import sparse
from scipy.linalg import fractional_matrix_power
import time

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import random
import argparse
import warnings
from math import log2
from sklearn.decomposition import PCA
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy import spatial
from scipy import sparse as sp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import phate
import scprep
import datetime
from scipy.spatial.distance import cdist, pdist
from scipy import sparse
from scipy.linalg import fractional_matrix_power
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from create_dataset_sense import plot_samples, load_data
import numpy as np
import pylab
import torch
from rnaseq_datasets import get_rnaseq_dataset
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
import numba
import sys
from matplotlib import cm
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
import warnings
import os
from numba.core.errors import NumbaWarning
sys.path.append(os.path.join(os.path.dirname(__file__), "contrastive-ne-master", "src", "cne"))
import main_cne 
sys.path.append('./umap')
import umap.plot
from umap.utils import submatrix, average_nn_distance
from snc.snc import SNC
warnings.filterwarnings("ignore", category=NumbaWarning)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#Function for constructing Incomplete Distance Matrix for PPDA

def dist_approx(X_na,X_a):

  DA = cdist(X_a,X_a, metric='euclidean')                   #anchor to anchor distance
  DNA = cdist(X_na,X_a, metric='euclidean')                 #non-anchor to anchor distance 
  
  #for k in attackers:
      #DNA[k] += np.full(DNA[k].shape, 2)
#       DNA[k] += 0.3*np.random.uniform(0,1,DNA[k].shape)
  #Distances with uniform random noise
  #DNA = cdist(X_na,X_a, metric='euclidean') + 0.1*np.random.uniform(0,1,(X_na.shape[0], X_a.shape[0]))

#   #Distance with LDP Noise
#   delta = 0.2
#   for i in range(DNA.shape[0]):
#     DNA[i] = DNA[i] * delta / (np.linalg.norm(DNA[i]))
#     DNA[i] += np.random.laplace(loc = 1, scale = 0.05, size = DNA[i].shape)
#   for i in range(DA.shape[0]):
#     DA[i] = DA[i]*delta / (np.linalg.norm(DA[i]))
  zero_mat = np.eye((X_na.shape[0]))
  ones_mat = np.ones((X_a.shape[0],X_na.shape[0]))
  ones_mat2 =  np.ones((X_a.shape[0],X_a.shape[0]))
  D = np.array(np.vstack((np.hstack((np.zeros((X_na.shape[0],X_na.shape[0])), DNA)), np.hstack((DNA.T, DA)))))    #Incomplete distance matrix
  
  W_1 = np.array(np.vstack((np.hstack((zero_mat,ones_mat.T)), np.hstack((ones_mat, ones_mat2)))))          #Weight matrix


  V = np.array(np.diag(np.matmul(W_1,np.ones(W_1.shape[0]))) - W_1)      #V matrix required for SMACOF
 

  V1=V[:X_na.shape[0],:X_na.shape[0]]
  V2=V[:X_na.shape[0],X_na.shape[0]:]

  return D, V, V1, V2, W_1, DA, DNA

#Function for learning embeddings through MDS

def classical_MDS_X(D, V, W_1, n,d):

  epsilon= 1e-3
 
  #D_inv = np.reciprocal(D,  out = np.zeros_like(D), where=(D!=0))

  #L_D_inv = np.diag(np.matmul(D_inv,np.ones(D_inv.shape[0]))) - D_inv
  #print(L_D_inv) 
  #Zu = np.random.multivariate_normal(np.zeros(n), np.linalg.pinv(L_D_inv), 50).T
  np.random.seed(10)  
  Zu = np.random.normal(0,1,(n, 500))               #Initializing Embeddings
  epochs = 2000
  loss = []
  V_inv = np.linalg.pinv(V)

  W_new = np.multiply(W_1,D)
  print(W_new)

  D_new = cdist(Zu,Zu, metric='euclidean')
    
  #SMACOF implementation

  for t in tqdm(range(epochs)):
    
   
    W_final = np.divide(W_new,D_new, out = np.zeros_like(W_new), where=(D_new!=0))
    B_Z = np.diag(np.matmul(W_final,np.ones(W_final.shape[0]))) - W_final 
    X_final = np.matmul(np.matmul(V_inv, B_Z), Zu)
    D_new = cdist(X_final,X_final, metric='euclidean')
    D_inv_new = np.reciprocal(D_new ,  out = np.zeros_like(D_new), where=(D_new!=0))
    W_upper_triag = np.array(D_inv_new[np.triu_indices(D_inv_new.shape[0], k = 1)])
    C = np.square(D - D_new)
    D_upper_triag = C[np.triu_indices(C.shape[0], k = 1)]
    stress = np.dot(W_upper_triag, D_upper_triag)
    loss.append(stress)
    Zu = X_final
    if t % 10 == 0:
       print(stress)
    
    if t!=0:
      if abs(loss[t]-loss[t-1]) < epsilon:
        break
    
  return X_final, loss

#Function for learning embeddings through Anchored-MDS 

def MDS_X(D, V1, V2, W_1, DA,X_na, X_a, n,d):
    
  print(D)
  D_inv = np.reciprocal(D,  out = np.zeros_like(D), where=(D!=0))

  L_D_inv = np.diag(np.matmul(D_inv,np.ones(D_inv.shape[0]))) - D_inv

  np.random.seed(32)
  Zu_samples = np.random.multivariate_normal(np.zeros(X_na.shape[0] + X_a.shape[0]), np.linalg.pinv(L_D_inv),d).T
  Zu = Zu_samples[:n,:]                    #Intializing Embeddings


  
  epsilon= 1e-3
  epochs = 2000
  loss = []
  V1_inv = np.linalg.pinv(V1)

  W_new = np.multiply(W_1,D)

  DNA_new = cdist(Zu,X_a, metric='euclidean')
  D_new = np.array(np.vstack((np.hstack((np.zeros((X_na.shape[0],X_na.shape[0])), DNA_new)), np.hstack((DNA_new.T, DA)))))
  for t in tqdm(range(epochs)):
   
    W_final = np.divide(W_new,D_new, out = np.zeros_like(W_new), where=(D_new!=0))

    B_Z = np.diag(np.matmul(W_final,np.ones(W_final.shape[0]))) - W_final 

    BZ1 = B_Z[:X_na.shape[0],:X_na.shape[0]]
    BZ2 = B_Z[:X_na.shape[0],X_na.shape[0]:]

    term1 = np.matmul(BZ1,Zu)
    term2_temp = BZ2 - V2
    term2 = np.matmul(term2_temp, X_a)


    X_final = np.matmul(V1_inv,(term1 + term2))

    DNA_new = cdist(X_final,X_a, metric='euclidean')   
    D_new = np.array(np.vstack((np.hstack((np.zeros((X_na.shape[0],X_na.shape[0])), DNA_new)), np.hstack((np.transpose(DNA_new), DA)))))
    D_inv_new = np.reciprocal(D_new ,  out = np.zeros_like(D_new), where=(D_new!=0))
    W_upper_triag = np.array(D_inv_new[np.triu_indices(D_inv.shape[0], k = 1)])

    C = np.square(D - D_new)
    
    D_upper_triag = C[np.triu_indices(C.shape[0], k = 1)]

    
    stress = np.dot(W_upper_triag, D_upper_triag)

    loss.append(stress)
    Zu = X_final
    if t % 10 == 0:
       print(stress)
    
    if t!=0:
      if abs(loss[t]-loss[t-1]) < epsilon:
        break
    
  return X_final, loss

#Function for computing error in estimation of distances
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

def dist_error(X_na, X_final):
  
  D_true = cdist(X_na, X_na, metric='euclidean')
  z_true = spatial.distance.squareform(D_true)

  D_esti = cdist(X_final, X_final, metric='euclidean')
  z_esti = spatial.distance.squareform(D_esti) 
  Error = np.linalg.norm((D_true - D_esti), 'fro')/ np.linalg.norm((D_true), 'fro')
  
  return Error, D_true, D_esti, z_true, z_esti

#Function for checking F-score for neighborhood structure preservation, where k is the number of nearest neighbor
#Input (k+1)  for k-NN as we are not considering distance of node from itself.

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
        #print("Node:{}, newlist1:{}, newlist2:{}".format(i+1, newlist1, newlist2))
        #print("Relative F-score for node: {} = {}".format(i+1, f_score))
      avg_f_score = sum(f_scores)/len(f_scores)

      return avg_f_score
    
#Function for checking similarity between graph structures obtained in non-private and private manner

def check_F_score(A_esti, A_org):
      temp1 = spatial.distance.squareform(A_esti)
      temp2 = spatial.distance.squareform(A_org)
      print(temp2)
      print(temp1)
      TP = 0
      FP = 0
      FN = 0
      FP_elements = []
      TP_elements = []
      for i in range(temp1.shape[0]):
        if(temp2[i] > 0 and temp1[i] > 0):
          TP+=1
          TP_elements.append(temp1[i])
        elif(temp2[i] == 0 and temp1[i] > 0):
          FP+=1
          FP_elements.append(temp1[i])
        elif(temp2[i] > 0 and temp1[i] == 0):
          FN+=1
        
      print("TP: {}, FP: {}, FN: {}".format(TP, FP, FN))
      F_score = (2*TP)/(2*TP + FP + FN)

      return F_score

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
                                                            'segerstolpe', 'amb', 'tm', 'zheng', 'DermaMNIST', 'PneumoniaMNIST', 'RetinaMNIST', 'BreastMNIST', 'BloodMNIST','OrganCMNIST', 'OrganSMNIST', 'OrganMNIST3D', 'FractureMNIST3D', 'breast_cancer_uci','RDkit', 'cora', 'citeseer', 'pubmed', 'uci_data_taiwanese'], required=True, 
                                                            help='Name of the dataset to load.')
    parser.add_argument('--num_clients', type=int, default=10, help='Number of clients to use for the dataset.')
    parser.add_argument('--n_samples_per_node', type=int, default=None, help='Number of samples on each client.')
    parser.add_argument('--iid', action='store_true', help='Set to enable IID; omit for non-IID setting.')
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha')
    parser.add_argument('--data_directory', type=str, default='./Dataset', help='Directory where dataset files are located.')
    parser.add_argument('--num_data', type=int, default=1000, help='subdata to use from full dataset')
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
    dataset_dir = data_directory
        # Ensure directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    # Compose full log file path
    log_file = os.path.join(
        dataset_dir,
        f"results_log_{args.dataset_name}_{timestamp}.txt"
    )

    if args.output is None:
        args.output = f"./results/pointwise/{dataset_name}_anchorvis.png"

    mnist_dataset = datasets.MNIST(root='mnist_data', train=True, download=True, transform=transforms.ToTensor())

    full_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=False)
    images_full, labels_full = next(iter(full_loader))

    # Flatten images and convert to numpy
    images = images_full.numpy().reshape(-1, 28 * 28)  # shape: (60000, 784)
    labels = labels_full.numpy()

    # === Step 2: Random permutation and split ===
    num_total = len(images)
    num_anchors = 783  # change as needed
    assert num_anchors < num_total, "Too many anchors requested."

    indices = np.random.permutation(num_total)
    anchor_indices = indices[:num_anchors]
    non_anchor_indices = indices[num_anchors:]

    # === Step 3: Get anchor and non-anchor sets ===
    X_a = images[anchor_indices]
    y_a = labels[anchor_indices]

    X_na = images[non_anchor_indices]
    y_na = labels[non_anchor_indices]
    X_na = X_na[:1000,:]
    y_na = y_na[:1000]

    print("Anchor set shape (X_a):", X_a.shape)
    print("Non-anchor set shape (X_na):", X_na.shape)
    labels = y_na

    n = X_na.shape[0]
    d = X_na.shape[1]


    #D, V, V1, V2, W_1, DA, DNA = dist_approx(X_na.astype('float'), X_a.astype('float'), attackers_list)
    D, V, V1, V2, W_1, DA, DNA = dist_approx(X_na.astype('float'), X_a.astype('float'))
    X_final, loss = MDS_X(D, V1, V2, W_1, DA, X_na.astype('float'), X_a.astype('float'), n, d)
    error, D_true, D_esti, z_true, z_esti = dist_error(X_na.astype('float'), X_final)
#     # --- Print Final Scores ---
    print("Error in distance approximation: ", error)
    fscore = check_score(D_true, D_esti, 11)
    print("F-score: ", fscore)


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




