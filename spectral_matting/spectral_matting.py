# spectral_matting.py
import numpy as np
from skimage import io, color, segmentation
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import cKDTree
from heapq import heappush, heappop
import networkx as nx
from skimage.future import graph
from skimage.segmentation import slic
import warnings

def superpixel_graph(img, n_segments=100, compactness=10):
    """
    Build adjacency graph of superpixels using skimage SLIC + RAG
    Returns: rag (networkx graph), labels
    """
    labels = slic(img, n_segments=n_segments, compactness=compactness, start_label=0)
    rag = graph.rag_mean_color(img, labels)
    return rag, labels

def mst_segmentation(rag, labels, threshold):
    """
    From region adjacency graph (RAG), build MST and cut edges above threshold.
    rag: networkx graph with 'weight' attribute
    threshold: float - remove edges with weight > threshold
    """
    # convert rag to networkx graph with weight
    G = rag.copy()
    # compute MST
    T = nx.minimum_spanning_tree(G, weight='weight')
    # remove edges with weight > threshold
    to_remove = [(u,v) for u,v,d in T.edges(data=True) if d.get('weight', 1.0) > threshold]
    T.remove_edges_from(to_remove)
    # get connected components
    comps = list(nx.connected_components(T))
    label_map = {}
    for cid, comp in enumerate(comps):
        for seg_id in comp:
            label_map[seg_id] = cid
    # create final label map based on region labels
    out = np.zeros_like(labels)
    for i in np.unique(labels):
        if i in label_map:
            out[labels==i] = label_map[i]
        else:
            out[labels==i] = -1
    return out

def segment_image(img_path, n_segments=400, compactness=10, threshold=20.0):
    img = io.imread(img_path)
    if img.dtype != np.float64:
        img = img.astype(np.float64) / 255.0
    rag, labels = superpixel_graph(img, n_segments=n_segments, compactness=compactness)
    # ensure rag edges have 'weight' (distance between region colors)
    for u,v,data in rag.edges(data=True):
        # weight already set by rag_mean_color (default), otherwise compute:
        if 'weight' not in data:
            data['weight'] = np.linalg.norm(rag.nodes[u]['mean color'] - rag.nodes[v]['mean color'])
    seg = mst_segmentation(rag, labels, threshold)
    return seg
