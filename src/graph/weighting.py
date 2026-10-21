"""Pure functions for task-graph edge weighting."""


def combine_graph_similarity(hla_similarity, tissue_similarity, tissue_weight):
    """Combine HLA and tissue similarity while keeping their weights summing to 1."""
    if not 0.0 <= tissue_weight <= 1.0:
        raise ValueError(f"tissue_weight must be in [0, 1], got {tissue_weight}")
    hla_weight = 1.0 - tissue_weight
    return hla_weight * hla_similarity + tissue_weight * tissue_similarity
