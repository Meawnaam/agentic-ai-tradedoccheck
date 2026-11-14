"""Minimal stand-in for langgraph.graph.StateGraph.

Used ONLY when the `langgraph` package is not installed (e.g. offline test environments).
It implements the subset of the API used in graph.py with the same semantics:
nodes return partial state updates; keys annotated with a reducer (Annotated[list, operator.add])
are combined, other keys are overwritten; conditional edges pick the next node.
"""
from __future__ import annotations

import typing

START, END = "__start__", "__end__"


class StateGraph:
    def __init__(self, schema):
        self.schema = schema
        self.nodes: dict = {}
        self.edges: dict = {}
        self.cond: dict = {}

    def add_node(self, name, fn):
        self.nodes[name] = fn

    def add_edge(self, a, b):
        self.edges[a] = b

    def add_conditional_edges(self, source, path, path_map=None):
        self.cond[source] = (path, path_map)

    def set_entry_point(self, name):
        self.edges[START] = name

    def compile(self):
        return _Compiled(self)


class _Compiled:
    def __init__(self, g: StateGraph):
        self.g = g
        self.reducers = {}
        for key, hint in typing.get_type_hints(g.schema, include_extras=True).items():
            meta = getattr(hint, "__metadata__", ())
            if meta and callable(meta[0]):
                self.reducers[key] = meta[0]

    def _apply(self, state, update):
        for k, v in (update or {}).items():
            state[k] = self.reducers[k](state[k], v) if k in self.reducers and k in state else v

    def _next(self, node, state):
        if node in self.g.cond:
            fn, mapping = self.g.cond[node]
            key = fn(state)
            return mapping[key] if mapping else key
        return self.g.edges.get(node, END)

    def stream(self, state, config=None, stream_mode="values"):
        state = dict(state)
        limit = (config or {}).get("recursion_limit", 25)
        node = self.g.edges[START]
        if stream_mode == "values":
            yield dict(state)
        steps = 0
        while node != END:
            steps += 1
            if steps > limit:
                raise RecursionError(f"graph exceeded recursion_limit={limit}")
            update = self.g.nodes[node](state)
            self._apply(state, update)
            yield dict(state) if stream_mode == "values" else {node: update}
            node = self._next(node, state)

    def invoke(self, state, config=None):
        final = dict(state)
        for final in self.stream(state, config, stream_mode="values"):
            pass
        return final
