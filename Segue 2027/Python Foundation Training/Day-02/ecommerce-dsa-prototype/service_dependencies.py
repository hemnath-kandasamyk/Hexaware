"""
service_dependencies.py
=======================
Module 3 - Microservice deployment order using a directed graph, BFS
(Kahn's algorithm) and DFS topological sorting. Both are written by hand,
with no NetworkX.

Business problem
    Some services need others to be running first. We must deploy them in an
    order where every service comes AFTER everything it depends on, and we must
    detect circular dependencies, where no such order exists.

Direction of edges
    The user describes dependencies as ("Cart", "Pricing") = "Cart depends on Pricing".
    In the graph we draw the arrow  Pricing -> Cart  ("deploy Pricing BEFORE Cart").
    So graph["Pricing"] = ["Cart"] lists the services that are waiting for Pricing.
"""

import math
from collections import deque
from typing import Optional

DEFAULT_SERVICES = ["Cart", "Pricing", "Inventory", "Product Catalogue", "Warehouse"]
ALL_SERVICES = DEFAULT_SERVICES + ["Payment", "Shipping", "Notification"]

# (service, depends_on)
SAMPLE_DEPENDENCIES = [
    ("Cart", "Pricing"),
    ("Cart", "Inventory"),
    ("Pricing", "Product Catalogue"),
    ("Inventory", "Warehouse"),
]

# Same as above plus "Product Catalogue depends on Cart" -> circular dependency:
# Cart needs Pricing, Pricing needs Product Catalogue, Product Catalogue needs Cart.
CYCLIC_DEPENDENCIES = SAMPLE_DEPENDENCIES + [("Product Catalogue", "Cart")]


def build_graph(services: list, dependencies: list) -> dict:
    """
    Build the directed graph as an adjacency list.
    graph[A] = sorted list of services that must wait for A (arrows A -> B).
    Services are always sorted so results are repeatable. Duplicate rows are ignored.
    """
    nodes = set(services)
    for service, needs in dependencies:
        nodes.add(service)
        nodes.add(needs)

    adjacency = {node: set() for node in nodes}
    for service, needs in dependencies:
        adjacency[needs].add(service)

    return {node: sorted(adjacency[node]) for node in sorted(adjacency)}


# ----------------------------------------------------------------------
# BFS based topological sort: Kahn's algorithm
# ----------------------------------------------------------------------
def kahn_topological_sort(graph: dict) -> dict:
    """
    1. Count in-degree = how many unmet dependencies each service still has.
    2. Put every service with in-degree 0 into a queue (nothing blocks them).
    3. Take one from the queue, deploy it, and reduce the in-degree of services
       waiting for it. Any that reach 0 join the queue.
    4. If we could not deploy every service, the rest are stuck in a cycle.
    """
    in_degree = {node: 0 for node in graph}
    for node in graph:
        for dependent in graph[node]:
            in_degree[dependent] += 1

    queue = deque(sorted(node for node in graph if in_degree[node] == 0))
    order, steps = [], []

    while queue:
        queue_before = ", ".join(queue)
        node = queue.popleft()
        order.append(node)

        unlocked = []
        for dependent in graph[node]:
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
                unlocked.append(dependent)

        steps.append({
            "Step": len(order),
            "Queue before": queue_before,
            "Deploy now": node,
            "Newly unblocked": ", ".join(unlocked) if unlocked else "-",
            "Order so far": " -> ".join(order),
        })

    has_cycle = len(order) < len(graph)
    deployed = set(order)
    stuck = sorted(node for node in graph if node not in deployed) if has_cycle else []
    return {"order": order, "has_cycle": has_cycle, "stuck": stuck, "steps": steps}


# ----------------------------------------------------------------------
# DFS based topological sort (also detects cycles)
# ----------------------------------------------------------------------
def dfs_topological_sort(graph: dict) -> dict:
    """
    Visit a service, then (deeply) everything waiting for it. When a service is
    completely finished, push it onto `finished`. Reversing `finished` gives a
    valid deployment order.

    Colours: WHITE = not visited, GRAY = visiting right now, BLACK = finished.
    Reaching a GRAY service again means we walked in a circle -> cycle found.
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    colour = {node: WHITE for node in graph}
    finished, steps, path = [], [], []
    cycle = None  # becomes a list of services when a cycle is found

    def log(event: str, node: str):
        steps.append({"Step": len(steps) + 1, "Event": event, "Service": node,
                      "Finished so far": ", ".join(finished) if finished else "-"})

    def visit(node: str):
        nonlocal cycle
        colour[node] = GRAY
        path.append(node)
        log("Start visiting", node)

        for dependent in graph[node]:
            if cycle is not None:
                return
            if colour[dependent] == GRAY:           # back to a service still in progress
                cycle = path[path.index(dependent):] + [dependent]
                log("Cycle found! Already visiting", dependent)
                return
            if colour[dependent] == WHITE:
                visit(dependent)

        if cycle is not None:
            return
        colour[node] = BLACK
        path.pop()
        finished.append(node)
        log("Finished (all waiting services done)", node)

    for node in graph:
        if colour[node] == WHITE:
            visit(node)
            if cycle is not None:
                break

    order = [] if cycle is not None else list(reversed(finished))
    return {"order": order, "has_cycle": cycle is not None, "cycle": cycle, "steps": steps}


def find_cycle(graph: dict) -> Optional[list]:
    """Return one cycle like ['A', 'B', 'C', 'A'] (arrows A->B->C->A), or None."""
    return dfs_topological_sort(graph)["cycle"]


def cycle_edges(cycle: Optional[list]) -> set:
    """Turn ['A', 'B', 'C', 'A'] into {('A','B'), ('B','C'), ('C','A')}."""
    if not cycle:
        return set()
    return {(cycle[i], cycle[i + 1]) for i in range(len(cycle) - 1)}


def verify_order(graph: dict, order: list) -> bool:
    """True if every service appears after all the services it depends on."""
    if sorted(order) != sorted(graph):
        return False
    position = {node: index for index, node in enumerate(order)}
    return all(position[a] < position[b] for a in graph for b in graph[a])


# ----------------------------------------------------------------------
# Layout (where to draw each service)
# ----------------------------------------------------------------------
def compute_layout(graph: dict, has_cycle: bool) -> dict:
    """
    Returns {service: (x, y)}.
    - No cycle : layers from left to right (column = longest chain of dependencies).
    - With cycle: services placed on a circle so loops are easy to see.
    """
    nodes = list(graph)

    if has_cycle:
        positions = {}
        for index, node in enumerate(nodes):
            angle = math.pi / 2 + 2 * math.pi * index / len(nodes)
            positions[node] = (1.6 * math.cos(angle), math.sin(angle))
        return positions

    in_degree = {node: 0 for node in graph}
    for node in graph:
        for dependent in graph[node]:
            in_degree[dependent] += 1

    level = {node: 0 for node in graph}
    queue = deque(node for node in graph if in_degree[node] == 0)
    while queue:
        node = queue.popleft()
        for dependent in graph[node]:
            level[dependent] = max(level[dependent], level[node] + 1)
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    columns = {}
    for node in graph:
        columns.setdefault(level[node], []).append(node)

    positions = {}
    for column, group in columns.items():
        group.sort()
        for row, node in enumerate(group):
            positions[node] = (float(column), (len(group) - 1) / 2 - row)
    return positions
