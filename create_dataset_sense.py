import torch
print(torch.__version__)
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
import os
import numpy as np
from collections import defaultdict, Counter
import random
import os
import pickle
import torch
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
import os
import pickle as pkl
import sys
import networkx as nx
import torch.nn.functional as F
import json
from networkx.readwrite import json_graph
import pdb
# from scipy.sparse.linalg.eigen.arpack import eigsh
import re
from time import perf_counter
import tabulate
import sys
import gzip
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_swiss_roll
import os
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
# from tensorflow.keras.datasets import fashion_mnist
from collections import defaultdict
import medmnist
from medmnist import INFO, Evaluator
from medmnist import (
    DermaMNIST, PneumoniaMNIST, RetinaMNIST, BreastMNIST, BloodMNIST,
    OrganCMNIST, OrganSMNIST, OrganMNIST3D, FractureMNIST3D
)


def load_data(args, num_data, test_size, dataset_name, num_clients=None, n_samples_per_node = None, data_dir='', iid=False, shuffle=True, shuffle_digits=False, balanced=False):
    train_data = []
    test_data = []
    data_counts = {}

    # Vary test sizes for medical CBME datasets
    cbme_datasets = [
        'xin', 'baron_mouse', 'baron_human', 'muraro', 'segerstolpe',
        'amb', 'tm', 'zheng', 'baron_5000', 'darmanis', 'deng',
        'mECS', 'Kolod', 'PBMC'
    ]

    # CBME datasets go through medical_data_cbme with varying test_size (already passed in args.test_size)
    if dataset_name in cbme_datasets:
        return medical_data_cbme(
            args,
            dataset_name=dataset_name,
            data_dir=data_dir,
            num_data=num_data,
            num_clients=num_clients,
            test_size=test_size,
            shuffle=shuffle,
            n_samples_per_node=n_samples_per_node,
            iid=iid,
            alpha=0.5,
            balanced=balanced
        )

    if dataset_name == 'BRCA':
        feature_files = [f for f in os.listdir(data_dir) if f.startswith('X_') and f.endswith('.npy')]
        label_files = [f for f in os.listdir(data_dir) if f.startswith('y_') and f.endswith('.npy')]

        # Map using client identifiers (last two characters)
        feature_dict = {f[-6:-4]: f for f in feature_files}
        label_dict = {f[-6:-4]: f for f in label_files}

        # Find intersection of client keys
        matched_keys = sorted(list(set(feature_dict.keys()) & set(label_dict.keys())))

        # Adjust num_clients based on available data if needed
        if len(matched_keys) < num_clients:
            print(f"⚠️  Reducing num_clients from {num_clients} to {len(matched_keys)} due to limited files.")
            num_clients = len(matched_keys)

        client_ids = matched_keys[:num_clients]

        if not iid and len(client_ids) < num_clients:
            raise ValueError("Not enough feature or label files for the specified number of clients in non-IID mode.")

        all_features, all_labels = [], []

        for client_suffix in client_ids:
            features_path = os.path.join(data_dir, feature_dict[client_suffix])
            labels_path = os.path.join(data_dir, label_dict[client_suffix])

            features = np.load(features_path, allow_pickle=True)
            labels = np.load(labels_path, allow_pickle=True)

            features = features[:, 1:-1]  # Adjust based on BRCA data format
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features)

            all_features.append(features_scaled)
            all_labels.append(labels)

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        if iid:
            indices = np.arange(len(all_features))
            if shuffle:
                np.random.shuffle(indices)
            all_features = all_features[indices]
            all_labels = all_labels[indices]

            total_per_client = len(all_features) // num_clients
            for i in range(num_clients):
                start = i * total_per_client
                end = (i + 1) * total_per_client if i < num_clients - 1 else len(all_features)
                X = all_features[start:end]
                y = all_labels[start:end]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                train_data.append((X_train, y_train))
                test_data.append((X_test, y_test))
        else:
            from collections import defaultdict
            labels = all_labels
            features = all_features
            n_classes = len(np.unique(labels))
            client_indices = [[] for _ in range(num_clients)]

            data_per_class = defaultdict(list)
            for idx, label in enumerate(labels):
                data_per_class[label].append(idx)

            if balanced:
                for cls in range(n_classes):
                    cls_indices = data_per_class[cls]
                    if shuffle:
                        np.random.shuffle(cls_indices)
                    proportions = np.random.dirichlet(np.repeat(0.5, num_clients))
                    proportions = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
                    cls_split = np.split(cls_indices, proportions)
                    for client_id, split in enumerate(cls_split):
                        client_indices[client_id].extend(split.tolist())

                min_len = min(len(idx) for idx in client_indices)
                for i in range(num_clients):
                    idx = client_indices[i]
                    if shuffle:
                        np.random.shuffle(idx)
                    if len(idx) > min_len:
                        client_indices[i] = idx[:min_len]
                    else:
                        extra = np.random.choice(idx, min_len - len(idx), replace=True)
                        client_indices[i] = idx + extra.tolist()
            else:
                for cls in range(n_classes):
                    cls_indices = data_per_class[cls]
                    if shuffle:
                        np.random.shuffle(cls_indices)
                    proportions = np.random.dirichlet(np.repeat(0.5, num_clients))
                    proportions = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
                    cls_split = np.split(cls_indices, proportions)
                    for client_id, split in enumerate(cls_split):
                        client_indices[client_id].extend(split.tolist())

            for i in range(num_clients):
                idx = client_indices[i]
                if shuffle:
                    np.random.shuffle(idx)
                X = features[idx]
                y = labels[idx]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                train_data.append((X_train, y_train))
                test_data.append((X_test, y_test))


    elif dataset_name == 'MNIST':
        mnist_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transforms.ToTensor())

        full_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=False)
        images_full, labels_full = next(iter(full_loader))
        images = images_full.numpy().reshape(-1, 28 * 28)
        labels = labels_full.numpy()

        if num_data is not None:
            indices = np.random.permutation(len(images))[:num_data]
            images = images[indices]
            labels = labels[indices]

        if args.iid:
            train_data = iid_split_mnist(
                dataset=mnist_dataset, 
                nb_nodes=num_clients, 
                n_samples_per_node=None, 
                num_data=num_data, 
                shuffle=shuffle
            )
        else:
            train_data = non_iid_split_medmnist(
                (images, labels), num_clients, n_samples_per_node,
                alpha=args.alpha, batch_size=None,
                shuffle=shuffle, balanced=args.balanced
            )

        # === Test Data Preparation ===
        test_set = datasets.MNIST(data_dir, train=False, download=True, transform=transforms.ToTensor())
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=len(test_set), shuffle=False)
        images_test_mnist, labels_test_mnist = next(iter(test_loader))
        features_test = images_test_mnist.numpy().reshape(-1, 28 * 28)
        labels_test = labels_test_mnist.numpy()

        
        dim = features_test.shape[1]
        # n = dim-1
        features_test_scaled = features_test
        labels_test = labels_test

        test_data = []
        start = 0
        for i in range(num_clients):
            if i < len(train_data):
                X_train, y_train = train_data[i][0]
                if i < num_clients - 1:
                    new_test_data = (features_test_scaled[start:start+1, :], labels_test[start:start+1])
                    start += 1
                else:
                    new_test_data = (features_test_scaled[start:, :], labels_test[start:])
                test_data.append(new_test_data)
                train_data[i] = (X_train, y_train)
            else:
                print(f"Warning: No training data for client {i}")

        train_data = [(X, y) for (X, y) in train_data if X is not None and y is not None]
        # Compute total number of training and test samples across all clients
        total_train_samples = sum(len(client_data[0]) for client_data in train_data)
        total_test_samples = sum(len(client_data[0]) for client_data in test_data)

        print(f" Total training samples used across all clients: {total_train_samples}")
        print(f" Total test samples used across all clients: {total_test_samples}")
        # print("total anchors:", len(labels_test))


    elif dataset_name == 'rnaseq':
        train_data, test_data = get_rnaseq_dataset(args=args, dataset_name=dataset_name, folder_path=data_dir, isCent=False) 

    elif dataset_name == 'german_credit':
        train_data, test_data = get_german_credit_data(data_dir=data_dir, num_clients=num_clients, n_samples_per_node=n_samples_per_node, iid=False, shuffle=True, num_data=num_data, balanced=False)

    elif dataset_name == 'cifar10':
        train_data, test_data = get_cifar10_dataset(args=args, dataset_name=dataset_name, folder_path=data_dir, isCent=False)

    elif dataset_name in ['cora', 'citeseer', 'pubmed']:
        train_data, test_data = prepare_citation_data(args, dataset_name=dataset_name, data_dir=data_dir, num_clients=num_clients,  iid=False, n_samples_per_node=None, alpha=0.5)    

 
    elif dataset_name == 'fashionmnist':
        train_data, test_data = load_fashion_mnist_data(args, data_dir=data_dir, num_clients=num_clients, n_samples_per_node=n_samples_per_node, num_data=num_data, iid=False, shuffle=True, alpha=0.5)
    
    elif dataset_name in ['DermaMNIST', 'PneumoniaMNIST', 'RetinaMNIST', 'BreastMNIST', 'BloodMNIST','OrganCMNIST', 'OrganSMNIST', 'OrganMNIST3D', 'FractureMNIST3D']:
        train_data, test_data = get_medmnist_data(args, dataset_name=dataset_name, data_dir=data_dir, num_data=num_data, num_clients=num_clients, n_samples_per_node=None, shuffle=True)
    
    elif dataset_name == 'uci_data_taiwanese':
        train_data, test_data = uci_data_taiwanese(args, data_dir=data_dir, num_clients=num_clients, n_samples_per_node=None, balanced=False)

    elif dataset_name in ['Gowalla', 'Foursquare']:
        train_data, test_data = recommendation_data(args,
            filename=os.path.join(data_dir, dataset_name, 'total.txt'),
            checkin_file=os.path.join(data_dir, dataset_name, 'total.txt'),
            num_clients=num_clients,
            iid=False, use_user_labels=True
        )
    
    return train_data, test_data

def non_iid_split1(dataset, nb_nodes, n_samples_per_node, alpha, batch_size=None, shuffle=True, test_size=0.2, random_state=42):
  
    np.random.seed(random_state)
    prng = np.random.default_rng(random_state)
    torch.manual_seed(random_state)
    
    if hasattr(dataset, 'targets'):
        if isinstance(dataset.targets, torch.Tensor):
            targets = dataset.targets.numpy()
        elif isinstance(dataset.targets, list):
            targets = np.array(dataset.targets)
        else:
            targets = dataset.targets
    else:
        targets = []
        for _, target in dataset:
            targets.append(target)
        targets = np.array(targets)
    
    num_classes = len(set(targets))
    
    if hasattr(dataset, 'data'):
        if isinstance(dataset.data, torch.Tensor):
            data = dataset.data.numpy()
        else:
            data = dataset.data
    else:
        data = []
        for img, _ in dataset:
            if isinstance(img, torch.Tensor):
                data.append(img.numpy())
            else:
                data.append(img)
        data = np.array(data)
    
    min_required_samples_per_client = 10
    min_samples = 0
    
    idx_clients = [[] for _ in range(nb_nodes)]
    
    while min_samples < min_required_samples_per_client:
        idx_clients = [[] for _ in range(nb_nodes)]
        
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            if shuffle:
                prng.shuffle(idx_k)
            
            proportions = prng.dirichlet(np.repeat(alpha, nb_nodes))
            
            proportions = np.array(
                [
                    p * (len(idx_j) < len(data) / nb_nodes)
                    for p, idx_j in zip(proportions, idx_clients)
                ]
            )
            proportions = proportions / proportions.sum()
          
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_k_split = np.split(idx_k, proportions)
            
            idx_clients = [
                idx_j + idx.tolist() for idx_j, idx in zip(idx_clients, idx_k_split)
            ]
        
        min_samples = min([len(idx_j) for idx_j in idx_clients])
    
    data_splitted = []
    for i in range(nb_nodes):
        client_indices = idx_clients[i]
        
        if shuffle:
            prng.shuffle(client_indices)
        
        client_data = data[client_indices]
        client_targets = targets[client_indices]
        
        if len(client_data.shape) > 2 and client_data.shape[1:] == (28, 28):
            client_data = client_data.reshape(-1, 28*28)
        
        # X_train, X_test, y_train, y_test = train_test_split(
        #     client_data, client_targets, test_size=0.0, random_state=random_state
        # )
        X_train, y_train = client_data, client_targets
        X_test, y_test = [], []
        data_splitted.append(((X_train, y_train), (X_test, y_test)))
        
        #data_splitted.append(((X_train, y_train), (X_test, y_test)))
    
    return data_splitted

def non_iid_split_mnist(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle, shuffle_digits=False):
    assert nb_nodes > 0 and nb_nodes <= 10
    nclasses = 10
    digits = torch.arange(nclasses) if not shuffle_digits else torch.randperm(nclasses, generator=torch.Generator().manual_seed(0))
    digits_split = []
    i = 0
    for n in range(nb_nodes, 0, -1):
        inc = int((nclasses - i) / n)
        digits_split.append(digits[i:i + inc])
        i += inc

    loader = torch.utils.data.DataLoader(dataset, batch_size=nb_nodes * n_samples_per_node, shuffle=shuffle)
    dataiter = iter(loader)
    images_train_mnist, labels_train_mnist = next(dataiter)

    data_splitted = []
    for i in range(nb_nodes):
        idx = torch.stack([y_ == labels_train_mnist for y_ in digits_split[i]]).sum(0).bool()
        features = images_train_mnist[idx].numpy().reshape(-1, 28 * 28)  # Flatten images
        labels = labels_train_mnist[idx].numpy()
        # scaler = MinMaxScaler()
        # features_scaled = scaler.fit_transform(features)

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        data_splitted.append(((X_train, y_train), (X_test, y_test)))

    return data_splitted


def iid_split_mnist(dataset, nb_nodes, n_samples_per_node=None, num_data=None, shuffle=True):
    all_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    all_images, all_labels = next(iter(all_loader))

    features = all_images.numpy().reshape(-1, 28 * 28)
    labels = all_labels.numpy()

    if num_data is not None:
        features = features[:num_data]
        labels = labels[:num_data]

    indices = np.arange(len(features))
    if shuffle:
        np.random.shuffle(indices)
        features = features[indices]
        labels = labels[indices]

    if n_samples_per_node is None:
        n_samples_per_node = len(features) // nb_nodes
        print(f"[INFO] Auto-setting n_samples_per_node to {n_samples_per_node}")

    total_required = nb_nodes * n_samples_per_node
    assert len(features) >= total_required, \
        f"Not enough data to split equally: need {total_required}, but got {len(features)}"

    features = features[:total_required]
    labels = labels[:total_required]

    data_splitted = []
    for i in range(nb_nodes):
        start = i * n_samples_per_node
        end = (i + 1) * n_samples_per_node
        X = features[start:end]
        y = labels[start:end]
        data_splitted.append(((X, y), None))

    return data_splitted


def non_iid_split_medmnist(dataset, nb_nodes, n_samples_per_node, alpha=0.1, batch_size=None, shuffle=True, balanced=True):
    images, labels = dataset
    labels = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    images = images.numpy() if isinstance(images, torch.Tensor) else images

    n_classes = len(np.unique(labels))
    total_samples = len(labels)

    client_indices = [[] for _ in range(nb_nodes)]

    if balanced:
        data_per_class = defaultdict(list)
        for idx, label in enumerate(labels):
            data_per_class[label].append(idx)

        for cls in range(n_classes):
            cls_indices = data_per_class[cls]
            if shuffle:
                np.random.shuffle(cls_indices)
            proportions = np.random.dirichlet(alpha=np.repeat(alpha, nb_nodes))
            proportions = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
            cls_split = np.split(cls_indices, proportions)
            for client_id, split in enumerate(cls_split):
                client_indices[client_id].extend(split.tolist())

        if n_samples_per_node is None:
            per_client = total_samples // nb_nodes
        else:
            per_client = n_samples_per_node

        for i in range(nb_nodes):
            indices = client_indices[i]
            if shuffle:
                np.random.shuffle(indices)
            if len(indices) >= per_client:
                client_indices[i] = indices[:per_client]
            else:
                extra = np.random.choice(indices, per_client - len(indices), replace=True)
                client_indices[i] = indices + extra.tolist()
    else:
        data_per_class = defaultdict(list)
        for idx, label in enumerate(labels):
            data_per_class[label].append(idx)

        for cls in range(n_classes):
            cls_indices = data_per_class[cls]
            if shuffle:
                np.random.shuffle(cls_indices)
            proportions = np.random.dirichlet(np.repeat(alpha, nb_nodes))
            proportions = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
            cls_split = np.split(cls_indices, proportions)
            for client_id, split in enumerate(cls_split):
                client_indices[client_id].extend(split.tolist())

        for i in range(nb_nodes):
            if len(client_indices[i]) < 30:
                extra = np.random.choice(client_indices[i], 30 - len(client_indices[i]), replace=True)
                client_indices[i].extend(extra.tolist())

    data_splitted = []
    for client_id in range(nb_nodes):
        indices = client_indices[client_id]
        if shuffle:
            np.random.shuffle(indices)
        subset_images = images[indices]
        subset_labels = labels[indices]

        if len(subset_images) == 0:
            print(f"Warning: Client {client_id} received 0 samples. Skipping.")
            continue

        features = subset_images.reshape(len(subset_images), -1)
        data_splitted.append(((features, subset_labels), None))

    return data_splitted

def iid_split_medmnist(dataset, nb_nodes, n_samples_per_node=None, shuffle=True):
    images, labels = dataset
    features = images.numpy().reshape(len(images), -1)
    labels = labels.numpy()

    if n_samples_per_node is None:
        n_samples_per_node = len(features) // nb_nodes
        print(f"[INFO] Auto-setting n_samples_per_node to {n_samples_per_node}")

    total_required = nb_nodes * n_samples_per_node
    assert len(features) >= total_required, f"Not enough data to split equally among all nodes. Required: {total_required}, Available: {len(features)}"

    if shuffle:
        indices = torch.randperm(len(features))[:total_required]
        features = features[indices]
        labels = labels[indices]
    else:
        features = features[:total_required]
        labels = labels[:total_required]

    data_splitted = []
    for i in range(nb_nodes):
        start = i * n_samples_per_node
        end = (i + 1) * n_samples_per_node
        X = features[start:end]
        y = labels[start:end]
        data_splitted.append(((X, y), None))

    return data_splitted


def read_node_label(filename):
    with open(filename, 'r') as f:
        return [line.strip().split('\t')[0] for line in f if line.strip()]

# === Get unique POIs from check-in file ===
def get_unique_pois(checkin_file):
    poi_set = set()
    with open(checkin_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                poi_id = parts[1].replace("LOC_", "")
                poi_set.add(int(poi_id))
    return sorted(poi_set)

# === Extract visualization labels: most frequent POI per user ===
def extract_user_labels(checkin_file):
    user_poi_counts = defaultdict(list)
    with open(checkin_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                user_id = parts[0]
                poi_id = int(parts[1].replace("LOC_", ""))
                user_poi_counts[user_id].append(poi_id)

    user_labels = {}
    for user, pois in user_poi_counts.items():
        most_common_poi = Counter(pois).most_common(1)[0][0]
        user_labels[user] = most_common_poi
    return user_labels

# === Split users into clients ===
def split_users_into_clients(user_list, num_clients=5, seed=42):
    np.random.seed(seed)
    user_list = np.random.permutation(user_list)
    return np.array_split(user_list, num_clients)

# === Simulate POI check-in data ===
def simulate_checkin_matrix(user_list, poi_list, min_checkins=5, max_checkins=20, 
                            seed=42, iid=True, client_id=0, num_clients=5):
    np.random.seed(seed)
    data = defaultdict(list)

    if not iid:
        poi_splits = np.array_split(poi_list, num_clients)
        local_pois = poi_splits[client_id]

    for user in user_list:
        n = np.random.randint(min_checkins, max_checkins)
        if iid:
            pois = np.random.choice(poi_list, size=n, replace=False)
        else:
            pois = np.random.choice(local_pois, size=n, replace=False)
        data[user] = pois.tolist()
    return data

# === Build (features, labels) per client ===
def build_client_data(users_split, poi_ids, user_labels=None, iid=True, num_clients=5, poi_dim=None):
    if poi_dim is None:
        poi_dim = max(poi_ids) + 1
    client_data = []
    for client_id, user_group in enumerate(users_split):
        checkin_data = simulate_checkin_matrix(user_group, poi_ids, iid=iid, 
                                               client_id=client_id, num_clients=num_clients)
        X_client, y_client = [], []
        for user in user_group:
            pois = checkin_data[user]
            vec = np.zeros(poi_dim)
            for poi in pois:
                vec[poi] = 1
            X_client.append(vec)
            label = user_labels.get(user, 0) if user_labels else 0
            y_client.append(label)
        client_data.append(((np.array(X_client), np.array(y_client)), None))
    return client_data

# === Split into train/test per client ===
def split_train_test(client_data, test_ratio, seed=0):
    train_data, test_data = [], []
    for (X, y), _ in client_data:
        n = X.shape[0]
        np.random.seed(seed)
        indices = np.random.permutation(n)
        test_size = int(n * test_ratio)
        test_idx = indices[:test_size]
        train_idx = indices[test_size:]

        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        train_data.append(((X_train, y_train), None))
        test_data.append(((X_test, y_test), None))
    return train_data, test_data

# === Main driver ===
def recommendation_data(args, filename, checkin_file, num_clients=10, iid=True, use_user_labels=True):
    user_ids = list(set(read_node_label(filename)))
    poi_ids = get_unique_pois(checkin_file)
    poi_dim = max(poi_ids) + 1
    user_labels = extract_user_labels(checkin_file) if use_user_labels else None
    users_split = split_users_into_clients(user_ids, num_clients=num_clients)
    client_raw_data = build_client_data(users_split, poi_ids, user_labels=user_labels, iid=iid, num_clients=num_clients, poi_dim=poi_dim)
    train_data, test_data = split_train_test(client_raw_data, test_ratio=args.test_size)
    return train_data, test_data


def uci_data_taiwanese(args, data_dir, num_clients=10, n_samples_per_node=None, shuffle=True, balanced=True):
    # Load CSV with no headers
    data_path = os.path.join(data_dir, 'wdbc.data')
    df = pd.read_csv(data_path, header=None)

    # Separate ID, diagnosis, and features
    X = df.iloc[:, 2:].values  # 30 features
    y = df.iloc[:, 1].values   # diagnosis (M or B)

    # Encode labels: M = 1, B = 0
    y = LabelEncoder().fit_transform(y)

    # Normalize features
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    

    # Split like MedMNIST
    if args.iid:
        train_data = iid_split_medmnist((X_tensor, y_tensor), num_clients, n_samples_per_node, shuffle=shuffle)
    else:
        train_data = non_iid_split_medmnist((X_tensor, y_tensor), num_clients, n_samples_per_node, alpha=0.1, batch_size=None, shuffle=shuffle, balanced=True)

    # Build test set (like MedMNIST logic)
    features_test = X
    labels_test = y
    dim = features_test.shape[1]
    n = dim - 1
    features_test_scaled = features_test[:n, :]
    labels_test = labels_test[:n]

    test_data = []
    start = 0
    for i in range(num_clients):
        if i < len(train_data):
            X_train, y_train = train_data[i][0]
            if i < num_clients - 1:
                new_test_data = (features_test_scaled[start:start+1, :], labels_test[start:start+1])
                start += 1
            else:
                new_test_data = (features_test_scaled[start:n, :], labels_test[start:n])
            test_data.append(new_test_data)
            train_data[i] = (X_train, y_train)
        else:
            print(f"Warning: No training data for client {i}")

    train_data = [(X, y) for (X, y) in train_data if X is not None and y is not None]
    return train_data, test_data


def get_medmnist_data(args, dataset_name, data_dir, num_data, num_clients, n_samples_per_node, shuffle=True):
    transform = None if '3D' in dataset_name else transforms.ToTensor()

    dataset_dict = {
        'DermaMNIST': DermaMNIST,
        'PneumoniaMNIST': PneumoniaMNIST,
        'RetinaMNIST': RetinaMNIST,
        'BreastMNIST': BreastMNIST,
        'BloodMNIST': BloodMNIST,
        'OrganCMNIST': OrganCMNIST,
        'OrganSMNIST': OrganSMNIST,
        'OrganMNIST3D': OrganMNIST3D,
        'FractureMNIST3D': FractureMNIST3D
    }

    if dataset_name not in dataset_dict:
        raise ValueError(f"Unsupported MedMNIST dataset: {dataset_name}")

    selected_dataset = dataset_dict[dataset_name]

    train_dataset = selected_dataset(split='train', root=data_dir, download=True, transform=transform)
    test_dataset = selected_dataset(split='test', root=data_dir, download=True, transform=transform)

    if num_data is not None:
        indices = torch.randperm(len(train_dataset))[:num_data]
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    images = torch.stack([
        torch.from_numpy(train_dataset[i][0]) if isinstance(train_dataset[i][0], np.ndarray)
        else train_dataset[i][0] for i in range(len(train_dataset))
    ])
    labels = torch.tensor([int(train_dataset[i][1]) for i in range(len(train_dataset))])

    if args.iid:
        train_data = iid_split_medmnist((images, labels), num_clients, n_samples_per_node, shuffle=shuffle)
    else:
        train_data = non_iid_split_medmnist(
            (images, labels), num_clients, n_samples_per_node,
            alpha=args.alpha, batch_size=None,
            shuffle=shuffle, balanced=args.balanced
        )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    images_test, labels_test = next(iter(test_loader))
    features_test = images_test.view(images_test.shape[0], -1).numpy()
    labels_test = labels_test.numpy()

    features_test_scaled = features_test
    dim = features_test_scaled.shape[1]
    # n = dim - 1
    features_test_scaled = features_test_scaled
    labels_test = labels_test

    print("initial data:", len(train_data), len(features_test_scaled))

    test_data = []
    start = 0
    for i in range(num_clients):
        if i < len(train_data):
            X_train, y_train = train_data[i][0]
            if i < num_clients - 1:
                new_test_data = (features_test_scaled[start:start+1, :], labels_test[start:start+1])
                start += 1
            else:
                new_test_data = (features_test_scaled, labels_test)
            test_data.append(new_test_data)
            train_data[i] = (X_train, y_train)
        else:
            print(f"Warning: No training data for client {i}")

    train_data = [(X, y) for (X, y) in train_data if X is not None and y is not None]
    return train_data, test_data


def plot_samples(data, channel: int, title=None, plot_name="", n_examples=20):
    n_rows = int(n_examples / 5)
    plt.figure(figsize=(1 * n_rows, 1 * n_rows))
    if title:
        plt.suptitle(title)
    X, y = data
    for idx in range(n_examples):
        ax = plt.subplot(n_rows, 5, idx + 1)
        image = 255 - X[idx, channel].view((28, 28))
        ax.imshow(image, cmap='gist_gray')
        ax.axis("off")

    if plot_name != "":
        plt.savefig(f"plots/" + plot_name + ".png")

    plt.tight_layout()



def load_MACOSKO(num_data=None, seed=42):
    path_to_file = './Dataset/macosko_2015.pkl.gz'
    assert os.path.exists(path_to_file), f"File not found: {path_to_file}"
    
    with gzip.open(path_to_file, 'rb') as f:
        data = pickle.load(f)

    x = data["pca_50"].astype(np.float32)
    y = data["CellType1"].astype(str)

    # Optional subsampling
    if num_data is not None and num_data < len(x):
        np.random.seed(seed)
        idx = np.random.choice(len(x), size=num_data, replace=False)
        x = x[idx]
        y = y[idx]

    X_train, X_test, Y_train, Y_test = train_test_split(
        x, y, test_size=0.1, random_state=42
    )

    enc = OrdinalEncoder()
    Y_train = enc.fit_transform([[i] for i in Y_train]).flatten()
    Y_test = enc.transform([[i] for i in Y_test]).flatten()

    return X_train, X_test, Y_train, Y_test

def rnaseq_noniid(num_users, method="dir", num_data=None, alpha=0.3, seed=42, path=''):
    np.random.seed(seed)
    random.seed(seed)

    train_images, _, train_labels, _ = load_MACOSKO(num_data=num_data, seed=seed)

    n_classes = len(np.unique(train_labels))
    dataset = train_images
    labels = train_labels

    if num_data is None:
        num_data = len(labels)

    K = n_classes
    y_train = labels
    N = y_train.shape[0]
    net_dataidx_map = {}
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}

    min_size = 0
    while min_size < 10:  # ensures minimum data per client
        idx_batch = [[] for _ in range(num_users)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, num_users))
            proportions = np.array([p * (len(idx_j) < N / num_users) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_split = np.split(idx_k, proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, idx_split)]
        min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(num_users):
        np.random.shuffle(idx_batch[j])
        dict_users[j] = np.array(idx_batch[j])

    client_data = []
    client_labels = []
    for indices in dict_users.values():
        client_data.append(np.take(dataset, indices, axis=0))
        client_labels.append(np.take(labels, indices, axis=0))

    return client_data, client_labels, dict_users

def get_rnaseq_dataset(args, dataset_name, folder_path, isCent=False):
    if dataset_name != 'rnaseq':
        print(f'Dataset {dataset_name} not implemented yet...')
        sys.exit(0)

    X_train, X_test, Y_train, Y_test = load_MACOSKO(num_data=args.num_data, seed=args.seed)

    if isCent:
        return X_train, Y_train

    if not args.iid:
        client_data, client_labels, dict_users = rnaseq_noniid(
            num_users=args.num_clients,
            method="dir",
            num_data=len(X_train),
            alpha=args.alpha,
            seed=args.seed,
            path=folder_path
        )
    else:
        print("IID mode not implemented for RNASEQ.")
        sys.exit(0)

    train_data = [(np.array(client_data[i]), np.array(client_labels[i])) for i in range(len(client_data))]

    # Evenly distribute test samples to each client
    num_clients = args.num_clients
    test_indices = np.arange(len(X_test))
    test_splits = np.array_split(test_indices, num_clients)
    test_data = [(X_test[split], Y_test[split]) for split in test_splits]

    return train_data, test_data


def get_german_credit_data(data_dir=None, num_clients=10, n_samples_per_node=None, iid=False, shuffle=True, num_data=None, alpha=0.5, balanced=True):
    

    if data_dir is None:
        raise ValueError("data_dir must be provided to locate the .arff file.")

    arff_path = os.path.join(data_dir, "dataset_31_credit-g.arff")
    data, meta = arff.loadarff(arff_path)
    df = pd.DataFrame(data)

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.decode("utf-8")

    X = df.drop('class', axis=1)
    y = df['class']

    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    y = LabelEncoder().fit_transform(y)

    if num_data is not None:
        X = X.iloc[:num_data]
        y = y[:num_data]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_np = X_train.values
    y_train_np = np.array(y_train)

    if n_samples_per_node is None:
        n_samples_per_node = len(X_train_np) // num_clients

    train_data = []

    if iid:
        indices = np.arange(len(X_train_np))
        if shuffle:
            np.random.shuffle(indices)
        for i in range(num_clients):
            start = i * n_samples_per_node
            end = (i + 1) * n_samples_per_node if i < num_clients - 1 else len(X_train_np)
            idx = indices[start:end]
            train_data.append((X_train_np[idx], y_train_np[idx]))
    else:
        labels = y_train_np
        features = X_train_np
        n_classes = len(np.unique(labels))

        client_indices = [[] for _ in range(num_clients)]

        if balanced:
            data_per_class = defaultdict(list)
            for idx, label in enumerate(labels):
                data_per_class[label].append(idx)

            for cls in range(n_classes):
                cls_indices = data_per_class[cls]
                if shuffle:
                    np.random.shuffle(cls_indices)
                proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
                proportions = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
                cls_split = np.split(cls_indices, proportions)
                for client_id, split in enumerate(cls_split):
                    client_indices[client_id].extend(split.tolist())

            min_len = min(len(idx) for idx in client_indices)
            target_len = min_len if n_samples_per_node is None else n_samples_per_node
            for i in range(num_clients):
                idx = client_indices[i]
                if shuffle:
                    np.random.shuffle(idx)
                if len(idx) >= target_len:
                    client_indices[i] = idx[:target_len]
                else:
                    extra = np.random.choice(idx, target_len - len(idx), replace=True)
                    client_indices[i] = idx + extra.tolist()
        else:
            data_per_class = defaultdict(list)
            for idx, label in enumerate(labels):
                data_per_class[label].append(idx)

            for cls in range(n_classes):
                cls_indices = data_per_class[cls]
                if shuffle:
                    np.random.shuffle(cls_indices)
                proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
                proportions = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
                cls_split = np.split(cls_indices, proportions)
                for client_id, split in enumerate(cls_split):
                    client_indices[client_id].extend(split.tolist())

        for i in range(num_clients):
            idx = client_indices[i]
            if shuffle:
                np.random.shuffle(idx)
            train_data.append((features[idx], labels[idx]))

    n = X_test.shape[1] - 1  
    features_test_scaled = X_test.iloc[:n, :].values
    labels_test = y_test[:n]

    test_data = []
    start = 0
    for i in range(num_clients):
        if i < num_clients - 1:
            new_test_data = (features_test_scaled[start:start+1, :], labels_test[start:start+1])
            start += 1
        else:
            new_test_data = (features_test_scaled[start:, :], labels_test[start:])
        test_data.append(new_test_data)

    return train_data, test_data


def load_cifar10(test=False):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.CIFAR10(root='./data', train=not test, download=True, transform=transform)

    X = dataset.data.astype(np.float32) / 255.0  # Normalize to [0, 1]
    Y = np.array(dataset.targets)

    return X, Y


def cifar10_iid(num_users, seed, num_data=50000, path=None):
    X_train, Y_train = load_cifar10()
    X_train = X_train.reshape(X_train.shape[0], -1)

    # Trim to num_data
    X_train = X_train[:num_data]
    Y_train = Y_train[:num_data]

    num_items = int(num_data / num_users)
    all_indices = np.arange(num_data)
    np.random.seed(seed)
    np.random.shuffle(all_indices)

    client_data = []
    client_labels = []
    dict_users = {}

    for i in range(num_users):
        idx = all_indices[i*num_items:(i+1)*num_items]
        client_data.append(X_train[idx])
        client_labels.append(Y_train[idx])
        dict_users[i] = idx

    return client_data, client_labels, dict_users


def cifar10_noniid(num_users, method="dir", num_data=50000, alpha=0.5, seed=0, path=None):
    X_train, Y_train = load_cifar10()
    X_train = X_train.reshape(X_train.shape[0], -1)

    # Trim to num_data
    X_train = X_train[:num_data]
    Y_train = Y_train[:num_data]

    num_classes = 10
    np.random.seed(seed)

    # Get indices of each class
    class_indices = [np.where(Y_train == i)[0] for i in range(num_classes)]

    client_indices = [[] for _ in range(num_users)]

    if method == "dir":
        # Dirichlet sampling
        for c in range(num_classes):
            idx_c = class_indices[c]
            np.random.shuffle(idx_c)

            proportions = np.random.dirichlet(alpha=[alpha]*num_users)
            proportions = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]

            split_indices = np.split(idx_c, proportions)

            for i, split in enumerate(split_indices):
                client_indices[i].extend(split)

    client_data = []
    client_labels = []
    dict_users = {}

    for i in range(num_users):
        idx = np.array(client_indices[i])
        client_data.append(X_train[idx])
        client_labels.append(Y_train[idx])
        dict_users[i] = idx

    return client_data, client_labels, dict_users


def get_cifar10_dataset(args, dataset_name, folder_path, isCent=False):
    if dataset_name != 'cifar10':
        print(f'Dataset {dataset_name} not implemented yet...')
        sys.exit(0)

    # Load train and test data
    X_train, Y_train = load_cifar10()
    X_test, Y_test = load_cifar10(test=True)

    # Flatten the images
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # Truncate to args.num_data
    X_train = X_train[:args.num_data]
    Y_train = Y_train[:args.num_data]

    dim = X_train.shape[1]
    # n = dim - 1
    
    X_test = X_test
    Y_test = Y_test

    if isCent:
        return X_train, Y_train

    # Federated split
    if not args.iid:
        client_data, client_labels, dict_users = cifar10_noniid(
            num_users=args.num_clients,
            method="dir",
            num_data=args.num_data,
            alpha=args.alpha,
            seed=args.seed,
            path=folder_path
        )
    else:
        client_data, client_labels, dict_users = cifar10_iid(
            num_users=args.num_clients,
            seed=args.seed,
            num_data=args.num_data,
            path=folder_path
        )

    train_data = [(np.array(client_data[i]), np.array(client_labels[i])) for i in range(len(client_data))]

    num_clients = args.num_clients
    test_indices = np.arange(len(X_test))
    test_splits = np.array_split(test_indices, num_clients)
    test_data = [(X_test[split], Y_test[split]) for split in test_splits]

    return train_data, test_data

def data_preprocessing_scRNA_1(data_file, label_file, data_name, percentage_to_use = 0.005):   
    data_df = pd.read_csv(data_file, delimiter=',')
    labels_df = pd.read_csv(label_file, delimiter=',')

    data_df = data_df.drop(data_df.columns[0], axis=1)
    if data_name == 'amb':
        labels, label_indices = pd.factorize(labels_df['Class'])
    else:
        labels, label_indices = pd.factorize(labels_df['x'])

    node_labels = labels
    num_classes = torch.unique(torch.tensor(node_labels)).shape[0]

    return data_df, node_labels, num_classes    


def iid_split(X, y, num_clients, n_samples_per_node):
    n_samples = len(X)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    if n_samples_per_node is None:
        n_samples_per_node = n_samples // num_clients

    train_data = []
    for i in range(num_clients):
        start_idx = i * n_samples_per_node
        end_idx = (i + 1) * n_samples_per_node if i < num_clients - 1 else n_samples
        client_indices = indices[start_idx:end_idx]
        train_data.append((X[client_indices], y[client_indices]))

    return train_data


def noniid_split(X, y, num_clients, alpha=0.5, balanced=True, shuffle=True):
    n_samples = len(X)
    indices = np.arange(n_samples)
    labels = np.unique(y)
    client_indices = [[] for _ in range(num_clients)]

    if balanced:
        data_per_class = defaultdict(list)
        for idx, label in enumerate(y):
            data_per_class[label].append(idx)

        for cls in labels:
            cls_indices = data_per_class[cls]
            if shuffle:
                np.random.shuffle(cls_indices)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            split_points = (np.cumsum(proportions) * len(cls_indices)).astype(int)[:-1]
            cls_split = np.split(cls_indices, split_points)
            for client_id, split in enumerate(cls_split):
                client_indices[client_id].extend(split.tolist())

        min_len = min(len(indices) for indices in client_indices)
        for i in range(num_clients):
            idx = client_indices[i]
            if shuffle:
                np.random.shuffle(idx)
            if len(idx) > min_len:
                client_indices[i] = idx[:min_len]
            elif len(idx) < min_len:
                extra = np.random.choice(idx, min_len - len(idx), replace=True)
                client_indices[i] = idx + extra.tolist()
    else:
        for label in labels:
            label_indices = indices[y == label]
            if shuffle:
                np.random.shuffle(label_indices)
            proportions = np.random.dirichlet(np.ones(num_clients) * alpha)
            split_points = (np.cumsum(proportions) * len(label_indices)).astype(int)[:-1]
            label_splits = np.split(label_indices, split_points)
            for i in range(num_clients):
                client_indices[i].extend(label_splits[i].tolist())

        for i in range(num_clients):
            if len(client_indices[i]) < 30:
                extra = np.random.choice(client_indices[i], 30 - len(client_indices[i]), replace=True)
                client_indices[i].extend(extra.tolist())

    client_data = []
    for i in range(num_clients):
        idx = np.array(client_indices[i])
        client_data.append((X[idx], y[idx]))

    return client_data


def medical_data_cbme(args, dataset_name, data_dir, num_data, num_clients, shuffle=True, n_samples_per_node=None, test_size=None, iid=False, alpha=0.5, balanced=True):
    # Paths setup
    data_list, data_path, data_repo_path, label_path = [], {}, {}, {}

    # Dataset config
    def config_dataset(name, folder):
        data_list.append(name)
        base = os.path.join(data_dir, "scRNAseq_Benchmark_Datasets", folder)
        data_repo_path[name] = base
        data_path[name] = os.path.join(base, f"Filtered_{name}_data.csv")
        label_path[name] = os.path.join(base, "Labels.csv")

    if dataset_name in ["xin", "baron_mouse", "baron_human", "muraro", "segerstolpe", "amb", "tm", "zheng"]:
        name_folder_map = {
            "xin": ("Pancreatic_data/Xin", "Xin_HumanPancreas"),
            "baron_mouse": ("Pancreatic_data/Baron_Mouse", "MousePancreas"),
            "baron_human": ("Pancreatic_data/Baron_Human", "Baron_HumanPancreas"),
            "muraro": ("Pancreatic_data/Muraro", "Muraro_HumanPancreas"),
            "segerstolpe": ("Pancreatic_data/Segerstolpe", "Segerstolpe_HumanPancreas"),
            "amb": ("AMB", "mouse_allen_brain"),
            "tm": ("TM", "TM"),
            "zheng": ("Zheng", "68K_PBMC")
        }
        folder, tag = name_folder_map[dataset_name]
        config_dataset(dataset_name, folder.replace(f"_{tag}", ""))
        data_path[dataset_name] = os.path.join(data_repo_path[dataset_name], f"Filtered_{tag}_data.csv")
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported")

    # Load data
    for data_name in data_list:
        X, y, _ = data_preprocessing_scRNA_1(data_path[data_name], label_path[data_name], data_name)
        X = X.values

    if num_data is not None:
        X = X[:num_data]
        y = y[:num_data]
    
    print("total data before split ", X.shape)
    # Train-test split
    dim = X.shape[1]
    if dim > X.shape[0]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        test_size = 0.2
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)   

    # Determine train splits
    if iid:
        if n_samples_per_node is None:
            n_samples_per_node = len(X_train) // num_clients
        train_data = iid_split(X_train, y_train, num_clients, n_samples_per_node)
    else:
        train_data = noniid_split(X_train, y_train, num_clients, alpha, balanced=balanced, shuffle=shuffle)

    features_test_scaled = X_test
    labels_test = y_test

    test_data = []
    start = 0
    for i in range(num_clients):
        if i < num_clients - 1:
            test_data.append((features_test_scaled[start:start + 1, :], labels_test[start:start + 1]))
            start += 1
        else:
            test_data.append((features_test_scaled[start:, :], labels_test[start:]))

    return train_data, test_data

def aug_normalized_adjacency(adj):
    adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt).tocoo()

def load_data_citation(args, data_dir):
    data = load_data_nc(args, args.dataset_name,  data_dir, args.seed)
    adj_n = aug_normalized_adjacency(data['adj_train'])
    data['adj_train'] = sparse_mx_to_torch_sparse_tensor(adj_n)
    data['features'] = sparse_mx_to_torch_sparse_tensor(data['features'])
    return data

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def split_data(labels, test_prop, seed):
    np.random.seed(seed)
    nb_nodes = labels.shape[0]
    all_idx = np.arange(nb_nodes)
    pos_idx = labels.nonzero()[0]
    neg_idx = (1. - labels).nonzero()[0]
    np.random.shuffle(pos_idx)
    np.random.shuffle(neg_idx)
    pos_idx = pos_idx.tolist()
    neg_idx = neg_idx.tolist()
    nb_pos_neg = min(len(pos_idx), len(neg_idx))
    nb_test = round(test_prop * nb_pos_neg)
    idx_test_pos, idx_train_pos = pos_idx[:nb_test], pos_idx[nb_test:]
    idx_test_neg, idx_train_neg = neg_idx[:nb_test], neg_idx[nb_test:]
    return idx_test_pos + idx_test_neg, idx_train_pos + idx_train_neg

def load_data_nc(args,dataset_name, data_dir, seed):
    if dataset_name in ['cora', 'pubmed', 'citeseer']:
        adj, features, labels, idx_train, idx_val, idx_test = load_citation_data(
            args,dataset_name, data_dir, seed
        )
    
        idx_test, idx_train = split_data(labels, args.test_size, args.seed)

    labels = torch.LongTensor(labels)
    data = {'adj_train': adj, 'features': features, 'labels': labels, 'idx_train': idx_train, 'idx_test': idx_test}
    return data

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def load_citation_data(args, dataset_name, data_dir, seed):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(data_dir, "ind.{}.{}".format(dataset_name, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(data_dir, "ind.{}.test.index".format(dataset_name)))
    test_idx_range = np.sort(test_idx_reorder)
    
    if dataset_name == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    labels = np.argmax(labels, 1)

    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))
    idx_val = range(len(y), len(y) + 500)

    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_features(features):
    row_sum = np.array(features.sum(1))
    row_inv = np.power(row_sum, -1).flatten()
    row_inv[np.isinf(row_inv)] = 0.
    row_mat_inv = sp.diags(row_inv)
    return row_mat_inv.dot(features)


def one_hot_encode(labels):
    classes = np.unique(labels)
    class_dict = {c: i for i, c in enumerate(classes)}
    one_hot = np.zeros((len(labels), len(classes)))
    for i, label in enumerate(labels):
        one_hot[i, class_dict[label]] = 1
    return one_hot


def split_dirichlet_non_iid(labels, num_clients, alpha, n_samples_per_node=None, shuffle=True):
    label_distribution = defaultdict(list)
    for idx, label in enumerate(labels):
        label_distribution[label].append(idx)

    client_indices = [[] for _ in range(num_clients)]

    for label, idxs in label_distribution.items():
        if shuffle:
            np.random.shuffle(idxs)
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
        split = np.split(idxs, proportions)
        for client_id, part in enumerate(split):
            client_indices[client_id].extend(part.tolist())

    for i in range(num_clients):
        if n_samples_per_node is not None:
            if len(client_indices[i]) >= n_samples_per_node:
                client_indices[i] = client_indices[i][:n_samples_per_node]
            else:
                extra = np.random.choice(client_indices[i], n_samples_per_node - len(client_indices[i]), replace=True)
                client_indices[i].extend(extra.tolist())

    return client_indices


def split_iid(labels, num_clients, n_samples_per_node=None, shuffle=True):
    indices = np.arange(len(labels))
    if shuffle:
        np.random.shuffle(indices)

    total_samples = len(indices)
    if n_samples_per_node is None:
        n_samples_per_node = total_samples // num_clients

    client_indices = [
        indices[i * n_samples_per_node:(i + 1) * n_samples_per_node].tolist()
        for i in range(num_clients)
    ]
    return client_indices


def prepare_citation_data(args, dataset_name, data_dir, num_clients, alpha, iid=False, n_samples_per_node=None):
    dataset = dataset_name
    data = load_data_citation(args, data_dir)
    features = data['features'].to_dense().numpy()
    labels = data['labels'].numpy()

    print("shapes of citation data: features, labels", features.shape, len(labels))

    if iid:
        client_indices = split_iid(labels, num_clients, n_samples_per_node, shuffle=True)
    else:
        client_indices = split_dirichlet_non_iid(labels, num_clients, alpha, n_samples_per_node, shuffle=True)

    train_data = []
    for client_id in range(num_clients):
        idx = client_indices[client_id]
        client_features = features[idx]
        client_labels = labels[idx]

        # Ensure shapes are consistent
        client_features = np.asarray(client_features)
        client_labels = np.asarray(client_labels)

        if client_features.ndim == 1:
            client_features = client_features.reshape(1, -1)
        if client_labels.ndim == 0:
            client_labels = np.expand_dims(client_labels, 0)

        train_data.append((client_features, client_labels))

    # === Fix test data structure
    test_idx = data['idx_test']
    features_test = features[test_idx]
    labels_test = labels[test_idx]
    dim=features_test.shape[1]
    n = dim-1
    features_test = features_test[:n,:]
    labels_test = labels_test[:n]

    test_data = []
    for i in range(num_clients):
        if i < num_clients - 1:
            test_data.append((features_test[i:i + 1], labels_test[i:i + 1]))
        else:
            test_data.append((features_test, labels_test))

    return train_data, test_data




def load_fashion_mnist_data(args, data_dir, num_clients, iid=False, num_data=None, shuffle=True, n_samples_per_node=None, alpha=0.5):
    train_path = os.path.join(data_dir, 'fashionmnist', 'fashion-mnist_train.csv')
    test_path  = os.path.join(data_dir, 'fashionmnist', 'fashion-mnist_test.csv')

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_full = train_df.iloc[:, 1:].values.astype(np.uint8)
    y_train_full = train_df.iloc[:, 0].values.astype(np.uint8)

    X_test = test_df.iloc[:, 1:].values.astype(np.uint8)
    y_test = test_df.iloc[:, 0].values.astype(np.uint8)

    # Optional subsampling
    if num_data is not None:
        X_train_full = X_train_full[:num_data]
        y_train_full = y_train_full[:num_data]

    # Assign dimension-related variables
    dim = X_train_full.shape[1]
    # n = dim - 1

    # Truncate test set for quick validation
    X_test = X_test
    y_test = y_test

    if args.iid:
        n_samples = len(X_train_full)
        if n_samples_per_node is None:
            n_samples_per_node = n_samples // num_clients

        indices = np.arange(n_samples)
        if shuffle:
            np.random.shuffle(indices)

        train_data = []
        for i in range(num_clients):
            start_idx = i * n_samples_per_node
            end_idx = (i + 1) * n_samples_per_node if i < num_clients - 1 else n_samples
            client_indices = indices[start_idx:end_idx]
            train_data.append((X_train_full[client_indices], y_train_full[client_indices]))
    else:
        # Create a mock dataset class compatible with non_iid_split1
        class FashionDataset:
            def __init__(self, data, targets):
                self.data = data.reshape(-1, 28, 28)
                self.targets = targets

        dataset = FashionDataset(X_train_full, y_train_full)
        split_data = non_iid_split1(dataset, nb_nodes=num_clients, n_samples_per_node=n_samples_per_node, alpha=alpha, shuffle=shuffle)
        train_data = [client_train for client_train, _ in split_data]

    # Minimal test samples (same logic for both iid and non-iid)
    test_data = []
    start = 0
    for i in range(num_clients):
        if i < num_clients - 1:
            test_data.append((X_test[start:start+1], y_test[start:start+1]))
            start += 1
        else:
            test_data.append((X_test[start:], y_test[start:]))

    return train_data, test_data
