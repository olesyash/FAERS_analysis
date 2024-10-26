import os
import urllib
from datetime import timedelta
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd
from cachier import cachier
import tempfile
import matplotlib.pyplot as plt
from collections import Counter

dir_this = os.path.dirname(os.path.abspath(__file__))
dir_data = os.path.join(dir_this, "MLConnectedWorldBook", "data")


def get_rabbi_quotation_data() -> nx.DiGraph:
    fn = os.path.join(dir_data, "rabbi_quotes.csv")
    assert os.path.exists(fn), f"File {fn} not found"
    df = pd.read_csv(fn)[
        [
            "first_rabbi_after_link",
            "second_rabbi_after_link",
            "first_rabbi_after_link_id",
            "second_rabbi_after_link_id",
            "token_string",
        ]
    ]
    df_edges = (
        df[["first_rabbi_after_link", "second_rabbi_after_link"]]
        .value_counts(normalize=True)
        .reset_index()
    )
    df_edges.columns = ["from", "to", "weight"]
    G = nx.from_pandas_edgelist(
        df_edges,
        "from",
        "to",
        "weight",
        create_using=nx.DiGraph,
    )
    G.name = "Rabbi quotation network"
    return G


def get_graph(dataset_name: str) -> Union[nx.Graph, list]:
    """Load a graph from a data collection repository or list available datasets."""

    data_names = {
        "ca-AstroPh": "ca-AstroPh.txt.gz",
        "ca-CondMat": "ca-CondMat.txt.gz",
        "ca-GrQc": "ca-GrQc.txt.gz",
        "ca-HepPh": "ca-HepPh.txt.gz",
        "ca-HepTh": "ca-HepTh.txt.gz",
        "rabbi_quotation_data": get_rabbi_quotation_data,
    }

    if dataset_name == "list":
        return list(data_names.keys())

    if callable(data_names.get(dataset_name, None)):
        return data_names[dataset_name]()

    try:
        return load_graph_from_local(dataset_name)
    except FileNotFoundError:
        pass

    if dataset_name not in data_names:
        raise ValueError(
            f"Dataset {dataset_name} not available. Choose from {list(data_names.keys())}."
        )
    return load_graph_from_web(data_names[dataset_name])


def load_graph_from_local(dataset_name: str) -> nx.Graph:
    """Load a graph from a local file."""

    for extension in ["", ".csv", ".csv.gz"]:
        dataset_path = os.path.join(dir_data, dataset_name + extension)
        if os.path.exists(dataset_path):
            directed = False
            for d in ("directed", "directional"):
                if d in dataset_name:
                    directed = True
                    break
            df = pd.read_csv(dataset_path)
            assert "src" in df.columns, f"Column 'src' not found in {dataset_path}"
            assert "dst" in df.columns, f"Column 'dst' not found in {dataset_path}"
            edge_attr = [c for c in df.columns if c not in ["src", "dst"]]
            G = nx.from_pandas_edgelist(
                df,
                source="src",
                target="dst",
                edge_attr=edge_attr,
                create_using=nx.DiGraph if directed else nx.Graph,
            )
            return G
    raise FileNotFoundError("Local dataset file not found")


@cachier(stale_after=timedelta(days=100))
def load_graph_from_web(filename: str) -> nx.Graph:
    """Download and load a graph from the web."""
    url_base = "https://snap.stanford.edu/data/"
    url = url_base + filename
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = os.path.join(tmpdirname, "graph.txt.gz")
        urllib.request.urlretrieve(url, fn)
        G = nx.read_edgelist(fn, nodetype=int)
    return G


@cachier(stale_after=timedelta(days=100))
def load_dataset_from_web(filename: str, sep="\t") -> pd.DataFrame:
    """Download and load a dataset from the web."""
    url_base = "https://snap.stanford.edu/data/"
    url = url_base + filename
    with tempfile.TemporaryDirectory() as tmpdirname:
        fn = os.path.join(tmpdirname, "dataset.csv.gz")
        urllib.request.urlretrieve(url, fn)
        df = pd.read_csv(fn, comment="#", sep=sep, header=None)
    return df


def load_dataset_from_local(dataset_name: str) -> pd.DataFrame:
    """Load a dataset from a local file."""

    tried = []
    for extension in ["", ".csv", ".csv.gz"]:
        dataset_path = os.path.join(dir_data, dataset_name + extension)
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            return df
        tried.append(dataset_path)
    # try pickle
    for extension in [".pkl", ".pkl.gz"]:
        dataset_path = os.path.join(dir_data, dataset_name + extension)
        if os.path.exists(dataset_path):
            df = pd.read_pickle(dataset_path)
            return df
        tried.append(dataset_path)
    raise FileNotFoundError("Local dataset file not found. Tried: " + ", ".join(tried))


def get_info(G: nx.Graph):
    """Get info about a graph"""
    ret = []
    ret.append(f"Name: {G.name}. Directed: {G.is_directed()}")
    ret.append(f"Number of nodes: {G.number_of_nodes():,d}")
    ret.append(f"Number of edges: {G.number_of_edges():,d}")
    ret.append(f"Average clustering: {nx.average_clustering(G)}")
    if nx.is_connected(G):
        ret.append(
            f"Average shortest path length: {nx.average_shortest_path_length(G)}"
        )
    else:
        ret.append("Graph is not connected")
    return "\n".join(ret)


def get_connected_component_subgraphs(G: nx.Graph) -> list:
    """Get connected components of a graph"""
    if nx.is_directed(G):
        tmp = G.to_undirected()
    else:
        tmp = G
    components = [G.subgraph(c).copy() for c in nx.connected_components(tmp)]
    # sort by node count
    components.sort(key=lambda x: x.number_of_nodes(), reverse=True)
    for i in range(len(components)):
        components[i].name = f"Component {i+1} of {G.name}"
    return components


def plot_degree_distribiution(
        G: nx.Graph,
        name:str,
        color:str='C0',
        ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    degrees = [d for n, d in G.degree()]
    degree_counts = np.array([[k, v] for k, v in sorted(Counter(degrees).items())])
    degree_counts = np.log(degree_counts)
    ax.plot(degree_counts[:, 0], degree_counts[:, 1], '.', color=color, label=name)
    ax.set_xlabel('log(degree)')
    ax.set_ylabel('log(Count)')
    x = degree_counts[0, 0] - 0.1
    y = degree_counts[0, 1]
    # linear fit
    m, b = np.polyfit(degree_counts[:, 0], degree_counts[:, 1], 1)
    xx = np.array([degree_counts[0, 0], degree_counts[-1, 0]])
    yy = m * xx + b
    ax.plot(xx, yy, color=color)
    ax.text(x, yy[0], f'{name} $\\alpha={m:.1f}$', color=color, ha='right', va='center')
    return ax