import math
import warnings
from datetime import timedelta
from io import BytesIO
from itertools import combinations
from typing import Literal

import matplotlib.offsetbox as offsetbox
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from cachier import cachier
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


@cachier(stale_after=timedelta(days=100))
def create_triad_image(triad_name, edges):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    G = nx.DiGraph()
    G.add_nodes_from([1, 2, 3])
    G.add_edges_from(edges)
    layout = {1: (0, 0), 2: (1, 0), 3: (0.5, 1)}
    nx.draw(
        G,
        ax=ax,
        with_labels=False,
        node_color="brown",
        node_size=3500,
        edge_color="black",
        width=20,
        arrows=True,
        arrowsize=60,
        pos=layout,
    )
    ax.set_axis_off()
    ax.set_facecolor("none")
    fig.patch.set_alpha(0)
    # add margin around the plot
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return buf


# List of triad types for directed graphs with descriptions
triad_types = [
    ("003 - No edges", []),
    ("012 - Single directed edge", [(1, 2)]),
    ('102 - Two directed edges forming a "V"', [(1, 2), (3, 1)]),
    ("021D - Directed path", [(1, 2), (2, 3)]),
    ("021U - Common target", [(1, 2), (3, 2)]),
    ("021C - Cyclic triad", [(1, 2), (2, 3), (3, 1)]),
    ("111D - Directed triangle", [(1, 2), (2, 3), (3, 2)]),
    ("111U - Directed triangle", [(1, 2), (3, 2), (3, 1)]),
    ("030T - Transitive triad", [(1, 2), (1, 3), (2, 3)]),
    ("030C - Cyclic triad", [(1, 2), (2, 3), (3, 1)]),
    ("201 - Transitive triad", [(1, 2), (2, 3), (3, 1), (1, 3)]),
    ("120D - Directed triangle with mutual dyad", [(1, 2), (2, 3), (3, 2), (1, 3)]),
    ("120U - Directed triangle with mutual dyad", [(1, 2), (2, 3), (3, 1), (3, 2)]),
    ("120C - Cyclic triad", [(1, 2), (2, 3), (3, 1), (1, 3)]),
    (
        "210 - Transitive triad with extra edge",
        [(1, 2), (2, 3), (3, 1), (1, 3), (3, 2)],
    ),
    ("300 - Full triad", [(1, 2), (2, 3), (3, 1), (1, 3), (3, 2), (2, 1)]),
]

# Generate and cache triad images
for name, edges in triad_types:
    create_triad_image(name, edges)


# Function to plot triad census with images as labels
def plot_triad_census(
    graph: nx.DiGraph,
    sort_by: Literal["n", "type"] = "n",
    normalize: bool = False,
    top_n: int = None,
    label_scale=0.05,
    ignore_no_edges: bool = False,
    ignore_non_triads: bool = False,
    ax=None,
):

    # List of triad types for directed graphs with descriptions
    triad_types = [
        ("003 - No edges", []),
        ("012 - Single directed edge", [(1, 2)]),
        ('102 - Two directed edges forming a "V"', [(1, 2), (3, 1)]),
        ("021D - Directed path", [(1, 2), (2, 3)]),
        ("021U - Common target", [(1, 2), (3, 2)]),
        ("021C - Cyclic triad", [(1, 2), (2, 3), (3, 1)]),
        ("111D - Directed triangle", [(1, 2), (2, 3), (3, 2)]),
        ("111U - Directed triangle", [(1, 2), (3, 2), (3, 1)]),
        ("030T - Transitive triad", [(1, 2), (1, 3), (2, 3)]),
        ("030C - Cyclic triad", [(1, 2), (2, 3), (3, 1)]),
        ("201 - Transitive triad", [(1, 2), (2, 3), (3, 1), (1, 3)]),
        ("120D - Directed triangle with mutual dyad", [(1, 2), (2, 3), (3, 2), (1, 3)]),
        ("120U - Directed triangle with mutual dyad", [(1, 2), (2, 3), (3, 1), (3, 2)]),
        ("120C - Cyclic triad", [(1, 2), (2, 3), (3, 1), (1, 3)]),
        (
            "210 - Transitive triad with extra edge",
            [(1, 2), (2, 3), (3, 1), (1, 3), (3, 2)],
        ),
        ("300 - Full triad", [(1, 2), (2, 3), (3, 1), (1, 3), (3, 2), (2, 1)]),
    ]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8), dpi=120)

    census = nx.triadic_census(graph)
    census = pd.Series(census)
    if normalize:
        total = census.sum()
        census = census / total
    if sort_by == "n":
        census = census.sort_values(ascending=True)
    elif sort_by == "type":
        census = census.sort_index(ascending=True)

    if ignore_no_edges:
        census = census.drop("003")
    if ignore_non_triads:
        census = census.drop(
            [
                "003",
                "012",
                "102",
                "021D",
                "021U",
                "111D",
                "111U",
            ],
            errors="ignore",
        )

    if top_n:
        if sort_by == "type":
            msg = "top_n makes little sense when sorting by type"
            warnings.warn(msg)
        census = census.tail(top_n)
    bars = ax.barh(census.index, census.values)

    # Add images as labels
    for bar, triad_key in zip(bars, census.index):
        for name, edges in triad_types:
            if name.startswith(triad_key):
                buf = create_triad_image(name, edges)
                img = Image.open(buf)
                im = offsetbox.OffsetImage(img, zoom=label_scale)
                ab = offsetbox.AnnotationBbox(
                    im,
                    (bar.get_width(), bar.get_y() + bar.get_height() / 2),
                    xybox=(20, 0),
                    frameon=False,
                    xycoords="data",
                    boxcoords="offset points",
                    pad=0,
                )
                ax.add_artist(ab)

    if normalize:
        ax.set_xlabel("Proportion")
    else:
        ax.set_xlabel("Count")
    sns.despine(ax=ax, left=True, bottom=True)
    return ax


def get_triad_type(triad):
    """
    Determine the type of triad based on the number of edges in an undirected graph.

    Parameters:
    triad (networkx.Graph): The triad subgraph.

    Returns:
    str: The triad type.
    """
    edges = triad.number_of_edges()
    if edges == 0:
        return "003"
    elif edges == 1:
        return "012"
    elif edges == 2:
        return "102"
    elif edges == 3:
        return "300"
    return "Unknown"


def triadic_metrics_undirected(G):
    """
    Calculate triadic closure metrics and triad types for each node in an undirected graph.

    Parameters:
    G (networkx.Graph): The input graph.

    Returns:
    pd.DataFrame: A dataframe with the node id as an index and the following columns:
        - clustering_coefficient: The clustering coefficient of the node.
        - closed_triads: The number of closed triads (triangles) the node participates in.
        - open_triads: The number of open triads (two-edge paths) the node is part of.
        - triad_<type>: The count of each triad type the node is part of.
    """
    triad_labels = {
        "003": "No edges",
        "012": "Single edge",
        "102": 'Two edges forming a "V"',
        "300": "Full triad",
    }

    data = {
        "clustering_coefficient": [],
        "closed_triads": [],
        "open_triads": [],
        **{f"triad_{label}": [] for label in triad_labels.values()},
    }

    node_triads = {
        node: {label: 0 for label in triad_labels.values()} for node in G.nodes()
    }

    total_combinations = math.comb(G.number_of_nodes(), 3)

    for nodes in tqdm(
        combinations(G.nodes(), 3),
        total=total_combinations,
        desc="Calculating triads",
        leave=False,
    ):
        subgraph = G.subgraph(nodes)
        triad_type = get_triad_type(subgraph)
        if triad_type in triad_labels:
            label = triad_labels[triad_type]
            for node in nodes:
                node_triads[node][label] += 1

    for n in tqdm(
        G.nodes(),
        total=G.number_of_nodes(),
        desc="Calculating metrics",
        leave=False,
    ):
        clustering_coefficient = nx.clustering(G, n)
        closed_triads = 0
        open_triads = 0
        for neighbor in G.neighbors(n):
            for second_neighbor in G.neighbors(neighbor):
                if second_neighbor != n:
                    if G.has_edge(n, second_neighbor):
                        closed_triads += 1
                    else:
                        open_triads += 1

        data["clustering_coefficient"].append(clustering_coefficient)
        data["closed_triads"].append(closed_triads // 2)
        data["open_triads"].append(open_triads)
        for label in triad_labels.values():
            data[f"triad_{label}"].append(node_triads[n][label])

    df = pd.DataFrame(data, index=G.nodes())
    return df


if __name__ == "__main__":

    # generate an example directed graph
    graph_directed = nx.DiGraph()
    for _ in range(10):
        graph_directed.add_edge(*np.random.choice(range(10), 2, replace=False))

    fig, ax = plt.subplots(figsize=(12, 8), dpi=120)
    plot_triad_census(
        graph_directed, ax=ax, normalize=True, sort_by="n", top_n=3, label_scale=0.1
    )
    plt.show()
