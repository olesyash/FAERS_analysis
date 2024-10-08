import json
from collections import defaultdict
from datetime import timedelta

import networkx as nx
import numpy as np
import pandas as pd
from cachier import cachier
from yake import yake
from tqdm.auto import tqdm


@cachier(stale_after=timedelta(days=1))
def build_bi_partite_graph(
    df: pd.DataFrame,
    first_node_key,
    second_node_key,
    n_grams: int = 2,
    top: int = 10
) -> nx.Graph:

    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=n_grams,  # up to n-grams
        dedupLim=0.9,  # two words are considered the same if the similarity is higher than this
        dedupFunc="seqm",  # similarity function. Options are seqm, jaccard, cosine, levenshtein, jaro, jaro_winkler, monge_elkan, keywordset
        windowsSize=1,  # number of words in the window
        top=top,  # number of keywords to extract
        features=None,  # list of features to extract. If None, it uses all the features. Options are: ngram, sif, first, freq
    )
    graph = nx.Graph()
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Building graph", leave=False
    ):
        first_node = row[first_node_key]
        second_node = row[second_node_key]

        if first_node not in graph:
            graph.add_node(
                first_node, label=first_node, type=first_node_key, count=1
            )
        else:
            graph.nodes[first_node]["count"] += 1

        if second_node not in graph:
            graph.add_node(
                second_node, label=second_node, type=second_node_key, count=1
            )
        else:
            graph.nodes[second_node]["count"] += 1

        if graph.has_edge(first_node, second_node):
            graph.edges[first_node, second_node]["weight"] += 1
        else:
            graph.add_edge(first_node, second_node, weight=1)
    return graph

def remove_isolated_nodes(graph: nx.Graph):
    nodes_to_remove = [n for n in graph.nodes if graph.degree(n) == 0]
    ret = graph.copy()
    ret.remove_nodes_from(nodes_to_remove)
    return ret


def get_largest_connected_component_graph(graph: nx.Graph) -> nx.Graph:
    largest_connected_component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_connected_component).copy()


def cleanup_nodes_by_percentile(
    graph: nx.Graph,
    node_attr: str = "count",
    percentile: float = 1.0,
    node_type: str = None,
):
    if node_type:
        nodes = [n for n, d in graph.nodes(data=True) if d["type"] == node_type]
    else:
        nodes = list(graph.nodes)
    counts = np.array([graph.nodes[n][node_attr] for n in nodes])
    threshold = np.percentile(counts, 100 - percentile)
    nodes_to_remove = [n for n in nodes if graph.nodes[n][node_attr] < threshold]
    ret = graph.copy()
    ret.remove_nodes_from(nodes_to_remove)
    return ret
