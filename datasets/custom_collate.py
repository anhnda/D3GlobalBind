import random
from typing import Tuple, List
import numpy as np
import dgl
import torch

from commons.geometry_utils import random_rotation_translation
from commons.process_mols import lig_rec_graphs_to_complex_graph
import random


def graph_collate(batch):
    complex_graphs, ligs_coords, recs_coords, pockets_coords_lig, pockets_coords_rec, geometry_graph, complex_names, idx = map(
        list, zip(*batch))
    geometry_graph = dgl.batch(geometry_graph) if geometry_graph[0] != None else None
    return dgl.batch(
        complex_graphs), ligs_coords, recs_coords, pockets_coords_lig, pockets_coords_rec, geometry_graph, complex_names, idx


def graph_collate_revised_bk(batch):
    lig_graphs, rec_graphs, ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, complex_names, idx = map(
        list, zip(*batch))
    geometry_graph = dgl.batch(geometry_graph) if geometry_graph[0] != None else None
    return dgl.batch(lig_graphs), dgl.batch(
        rec_graphs), ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, complex_names, idx


def append_x(l1, l2):
    if type(l1) == list:
        return l1 + l2
    elif type(l1) == np.ndarray:
        return np.concatenate([l1, l2], axis=1)
    else:
        raise TypeError("Not define append for ", type(l1))


def get_sub(l, indices):
    if type(l) == np.ndarray:
        return l[indices]
    elif type(l) == list:
        return [l[i] for i in indices]
    else:
        raise TypeError("Not define get_sub of type: ", type(l))


def graph_collate_revised_negative_sampling(batch, fraction=1.0):
    lig_graphs, rec_graphs, ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, complex_names, idx = map(
        list, zip(*batch))
    batch_size = len(lig_graphs)
    neg_size = int(batch_size * fraction)
    indices = [i for i in range(batch_size)]
    neg_ids_lig = random.choices(indices, k=neg_size)
    neg_ids_rec = random.choices(indices, k=neg_size)
    lig_graphs_negs = get_sub(lig_graphs, neg_ids_lig)
    lig_coords_negs = get_sub(ligs_coords, neg_ids_lig)
    rec_graphs_negs = get_sub(rec_graphs, neg_ids_rec)
    recs_coords_negs = get_sub(recs_coords, neg_ids_rec)
    all_rec_coords_negs = get_sub(all_rec_coords, neg_ids_rec)
    pockets_coords_lig_negs = get_sub(pockets_coords_lig, neg_ids_lig)

    complex_names_negs = ["NEG_SAMPLE_%s" % i for i in range(neg_size)]

    if geometry_graph[0] != None:
        geometry_graph_negs = get_sub(geometry_graph, neg_ids_lig)
        geometry_graph = append_x(geometry_graph, geometry_graph_negs)
    assert len(all_rec_coords) == len(recs_coords)
    lig_graphs = append_x(lig_graphs, lig_graphs_negs)
    rec_graphs = append_x(rec_graphs, rec_graphs_negs)
    ligs_coords = append_x(ligs_coords, lig_coords_negs)
    recs_coords = append_x(recs_coords, recs_coords_negs)
    all_rec_coords = append_x(all_rec_coords, all_rec_coords_negs)
    pockets_coords_lig = append_x(pockets_coords_lig, pockets_coords_lig_negs)
    complex_names = append_x(complex_names, complex_names_negs)

    geometry_graph = dgl.batch(geometry_graph) if geometry_graph[0] != None else None
    return dgl.batch(lig_graphs), dgl.batch(
        rec_graphs), ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, complex_names, idx


def torsion_collate(batch):
    lig_graphs, rec_graphs, angles, masks, ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, complex_names, idx = map(
        list, zip(*batch))
    geometry_graph = torch.cat(geometry_graph, dim=0) if geometry_graph[0] != None else None
    return dgl.batch(lig_graphs), dgl.batch(rec_graphs), torch.cat(angles, dim=0), torch.cat(masks,
                                                                                             dim=0), ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, complex_names, idx


class AtomSubgraphCollate(object):
    def __init__(self, random_rec_atom_subgraph_radius=10):
        self.random_rec_atom_subgraph_radius = random_rec_atom_subgraph_radius

    def __call__(self, batch: List[Tuple]):
        lig_graphs, rec_graphs, ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, complex_names, idx = map(
            list, zip(*batch))

        rec_subgraphs = []
        for i, (lig_graph, rec_graph) in enumerate(zip(lig_graphs, rec_graphs)):
            rot_T, rot_b = random_rotation_translation(translation_distance=2)
            translated_lig_coords = ligs_coords[i] + rot_b
            min_distances, _ = torch.cdist(rec_graph.ndata['x'],
                                           translated_lig_coords.to(rec_graph.ndata['x'].device)).min(dim=1)
            rec_subgraphs.append(dgl.node_subgraph(rec_graph, min_distances < self.random_rec_atom_subgraph_radius))

        geometry_graph = dgl.batch(geometry_graph) if geometry_graph[0] != None else None

        return dgl.batch(lig_graphs), dgl.batch(
            rec_subgraphs), ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, complex_names, idx


class SubgraphAugmentationCollate(object):
    def __init__(self, min_shell_thickness=2):
        self.min_shell_thickness = min_shell_thickness

    def __call__(self, batch: List[Tuple]):
        lig_graphs, rec_graphs, ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, complex_names, idx = map(
            list, zip(*batch))

        rec_subgraphs = []
        for lig_graph, rec_graph in zip(lig_graphs, rec_graphs):
            lig_centroid = lig_graph.ndata['x'].mean(dim=0)
            distances = torch.norm(rec_graph.ndata['x'] - lig_centroid, dim=1)
            max_distance = torch.max(distances)
            min_distance = torch.min(distances)
            radius = min_distance + self.min_shell_thickness + random.random() * (
                        max_distance - min_distance - self.min_shell_thickness).abs()
            rec_subgraphs.append(dgl.node_subgraph(rec_graph, distances <= radius))
        geometry_graph = dgl.batch(geometry_graph) if geometry_graph[0] != None else None

        return dgl.batch(lig_graphs), dgl.batch(
            rec_subgraphs), ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, complex_names, idx
