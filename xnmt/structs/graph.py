import functools

from typing import List, Any, Dict
from collections import defaultdict


class HyperNode(object):
  """
  Represents a single HyperNode in a graph.
  - data: Value of the node
  - node_id: A unique id of the node.
  """
  def __init__(self, value: Any, node_id: int):
    self._value = value
    self._node_id = node_id

  @property
  def value(self):
    return self._value

  @property
  def node_id(self):
    return self._node_id

  def reversed(self):
    return self

  def feature_str(self):
    return ""

  def __repr__(self):
    return "Node({}, {})".format(self.node_id, str(self.value))


class HyperEdge(object):
  """
  Represents a single HyperEdge in a graph.
  - node_from: Source Node.
  - node_to: Destintation Nodes.
  - features: Float values representing the weight/features of the weight.
  """
  def __init__(self,
               node_from: int,
               node_to: List[int],
               features: List[float] = None,
               label: str = None):
    self._node_from = node_from
    self._node_to = tuple(node_to)
    self._features = tuple(features) if features is not None else features
    self._label = label

  @property
  def node_from(self):
    return self._node_from

  @property
  def node_to(self):
    return self._node_to

  @property
  def features(self):
    return self._features

  @property
  def label(self):
    return self._label

  def __repr__(self):
    return "Edge({} -> {})".format(self.node_from,
                                   str([child for child in self.node_to]))


class HyperGraph(object):
  """
  A hypergraph datastructure. Represented with a list of HyperEdge.
  - edge_list: The list of hyperedge forming the graph.
  - node_List: A map of node_id and the corresponding node
  """
  def __init__(self, edge_list: List[HyperEdge], node_list: Dict[int, HyperNode]):
    self._edge_list = tuple(edge_list)
    self._node_list = node_list
    succ_list, pred_list = self._build_graph()
    self._succ_list = succ_list
    self._pred_list = pred_list

  # If hypergraph is immutable, we can cache the reverse of the graph
  @functools.lru_cache(maxsize=1)
  def reverse(self):
    rev_edge_list = []
    for edge in self._edge_list:
      assert len(edge.node_to) == 1, "Does not support reversed of HyperGraph."
      rev_edge_list.append(HyperEdge(edge.node_to[0], [edge.node_from], edge.features, edge.label))
    node_list = {node_id: node.reversed() for node_id, node in self._node_list.items()}
    return HyperGraph(rev_edge_list, node_list)

  # If hypergraph is immutable, we can cache the topological sort of the graph
  @functools.lru_cache(maxsize=1)
  def topo_sort(self):
    # Buffers for topological sorting
    stack = []
    visited = [False for _ in range(len(self._node_list))]
    # Helper function for topological sorting
    def _topo_sort(current_id):
      visited[current_id] = True
      if current_id in self._succ_list:
        for adj_id, _ in self._succ_list[current_id]:
          if not visited[adj_id]:
            _topo_sort(adj_id)
      stack.append(current_id)
    # Driver function for topo sort
    for node_id in sorted(self._node_list.keys()):
      if not visited[node_id]:
        _topo_sort(node_id)
    # The results are seen from the reversed list
    return list(reversed(stack))

  def predecessors(self, node_id, with_edge=False):
    return self._with_edge(self._pred_list.get(node_id, []), with_edge)

  def sucessors(self, node_id, with_edge=False):
    return self._with_edge(self._succ_list.get(node_id, []), with_edge)

  def _with_edge(self, lst, with_edge):
    if with_edge:
      return lst
    else:
      return [node_id for node_id, _ in lst]

  # Leaves are nodes who have predecessors but no sucessors
  @functools.lru_cache(maxsize=1)
  def leaves(self):
    return sorted([x for x in self._pred_list if x not in self._succ_list])

  # Roots are nodes who have 0 predecessors
  @functools.lru_cache(maxsize=1)
  def roots(self):
    return sorted([x for x in self._node_list if x not in self._pred_list])

  def _build_graph(self):
    pred_list = defaultdict(list)
    succ_list = defaultdict(list)
    for edge in self._edge_list:
      from_id = edge.node_from
      assert from_id in self._node_list
      for to_id in edge.node_to:
        assert to_id in self._node_list
        succ_list[from_id].append((to_id, edge))
        pred_list[to_id].append((from_id, edge))
    return dict(succ_list), dict(pred_list)

  @property
  def len_nodes(self):
    return len(self._node_list)

  @property
  def len_edges(self):
    return len(self._edge_list)

  @functools.lru_cache(maxsize=1)
  def sorted_nodes(self):
    return sorted(self._node_list.values(), key=lambda x:x.node_id)

  def iter_nodes(self):
    return iter(self.sorted_nodes())

  def iter_edges(self):
    return iter(self._edge_list)

  def __repr__(self):
    lst = []
    for node in self._node_list.values():
      lst.append(repr(node))
    for edge in self._edge_list:
      lst.append(repr(edge))
    return "\n".join(lst)

  def __getitem__(self, node_id):
    return self._node_list[node_id]

