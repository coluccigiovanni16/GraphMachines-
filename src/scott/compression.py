"""
	Compression module
	===================
	
	Translations between fetchable and space-efficient graph representations.
"""

from .structs.cgraph import CGraph
from .structs.graph import Graph


def flatten(graph: Graph) -> CGraph:
    pass


def deflate(cgraph: CGraph) -> Graph:
    pass
