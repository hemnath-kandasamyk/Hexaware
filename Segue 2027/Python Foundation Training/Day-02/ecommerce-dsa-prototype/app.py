"""
app.py
======
Streamlit dashboard for the "E-Commerce Catalogue and Traffic Analysis" prototype.

This file contains ONLY user-interface code. All algorithms live in their own files:
    bst_catalogue.py, sku_hash_table.py, service_dependencies.py,
    traffic_analysis.py, gift_card_pairs.py

Run with:   streamlit run app.py
"""

import textwrap

import matplotlib

matplotlib.use("Agg")  # draw charts without needing a display window
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd
import streamlit as st

from bst_catalogue import CategoryBST, build_sample_tree, clean_name
from gift_card_pairs import (DEFAULT_ORDER_VALUES, DEFAULT_TARGET, brute_force_pairs,
                             is_sorted, parse_int_list, two_pointer_pairs, unsorted_demo)
from service_dependencies import (ALL_SERVICES, CYCLIC_DEPENDENCIES, DEFAULT_SERVICES,
                                  SAMPLE_DEPENDENCIES, build_graph, compute_layout,
                                  cycle_edges, dfs_topological_sort, find_cycle,
                                  kahn_topological_sort, verify_order)
from sku_hash_table import (SAMPLE_PRODUCTS, SMALL_TABLE_SIZE, SKUHashTable, hash_function,
                            hash_steps, run_benchmark)
from traffic_analysis import (DEFAULT_TRAFFIC, format_hour_range, generate_random_traffic,
                              naive_window_max, sliding_window_max)

# ----------------------------------------------------------------------
# Look and feel
# ----------------------------------------------------------------------
BLUE, AMBER, GREEN, RED = "#2563EB", "#F59E0B", "#16A34A", "#DC2626"
SOFT_BLUE, SOFT_AMBER = "#93C5FD", "#FEF3C7"
INK, MUTED, GRID, BG, WHITE = "#0F172A", "#64748B", "#CBD5E1", "#F8FAFC", "#FFFFFF"

CSS = """
<style>
.block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px;}
.hero {background: linear-gradient(135deg, #EFF6FF 0%, #FFFFFF 100%); border: 1px solid #DBEAFE;
       border-radius: 14px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; color: #0F172A;}
.hero b {color: #1D4ED8;}
.card {background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
       padding: 0.9rem 1.2rem 1rem 1.2rem; margin-bottom: 0.8rem; color: #0F172A;
       font-size: 0.95rem; line-height: 1.5;}
.card h4 {margin: 0 0 0.4rem 0; font-size: 0.78rem; letter-spacing: 0.06em;
          text-transform: uppercase; color: #2563EB;}
.card ul {margin: 0.2rem 0 0 1.1rem; padding: 0;}
</style>
"""

PAGES = [
    "1. Product Category BST",
    "2. SKU Hash Table",
    "3. Microservice Dependency Graph",
    "4. Traffic Analysis",
    "5. Gift Card Pair Finder",
    "Decision Table",
]


def card(title: str, body_html: str) -> None:
    st.markdown(f'<div class="card"><h4>{title}</h4>{body_html}</div>', unsafe_allow_html=True)


def intro_cards(problem: str, overview: str) -> None:
    left, right = st.columns(2)
    with left:
        card("Business problem", problem)
    with right:
        card("Algorithm overview", overview)


def closing_cards(complexity: str, real_world: str) -> None:
    left, right = st.columns(2)
    with left:
        card("Time and space complexity", complexity)
    with right:
        card("Real-world application", real_world)


def show_message(kind: str, text: str) -> None:
    {"success": st.success, "error": st.error, "warning": st.warning}.get(kind, st.info)(text)


def new_figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


def show_figure(fig) -> None:
    st.pyplot(fig)
    plt.close(fig)


# ======================================================================
# MODULE 1 - Binary Search Tree
# ======================================================================
def draw_bst(tree: CategoryBST, view: dict):
    """Draw the tree. Amber = nodes visited by the last operation, green = target node,
    numbered badges = visit order of the chosen traversal."""
    items = tree.layout()
    if not items:
        fig, ax = new_figure(6, 1.6)
        ax.text(0.5, 0.5, "The tree is empty", ha="center", va="center", color=MUTED)
        ax.axis("off")
        return fig

    count = len(items)
    max_depth = max(item["depth"] for item in items)
    fig, ax = new_figure(min(max(7.0, count * 0.8), 28.0), max(3.0, 1.15 * (max_depth + 1) + 0.5))
    ax.axis("off")

    position = {item["value"]: (item["x"], -item["depth"]) for item in items}
    path = set(view.get("path") or [])
    focus = view.get("focus")
    order = view.get("order") or {}

    for item in items:  # edges first, so boxes are drawn on top of them
        if item["parent"] is not None:
            px, py = position[item["parent"]]
            cx, cy = position[item["value"]]
            on_path = item["value"] in path and item["parent"] in path
            ax.plot([px, cx], [py, cy], color=AMBER if on_path else GRID,
                    lw=2.4 if on_path else 1.4, zorder=1)

    for item in items:
        name = item["value"]
        x, y = position[name]
        if name == focus:
            face, text_color, edge = GREEN, WHITE, GREEN
        elif name in path:
            face, text_color, edge = SOFT_AMBER, INK, AMBER
        else:
            face, text_color, edge = WHITE, INK, BLUE
        ax.text(x, y, textwrap.fill(name, 11), ha="center", va="center", fontsize=8.5,
                color=text_color, zorder=3,
                bbox=dict(boxstyle="round,pad=0.4", fc=face, ec=edge, lw=1.6))
        if name in order:
            ax.text(x, y + 0.38, str(order[name]), ha="center", va="center", fontsize=8,
                    color=WHITE, fontweight="bold", zorder=4,
                    bbox=dict(boxstyle="circle,pad=0.25", fc=BLUE, ec=BLUE))

    ax.set_xlim(-0.8, count - 0.2)
    ax.set_ylim(-max_depth - 0.7, 0.8)
    fig.tight_layout()
    return fig


def render_bst() -> None:
    ss = st.session_state
    if "bst" not in ss:
        ss.bst = build_sample_tree()
        ss.bst_msg = ("info", "Loaded the 15 sample categories. Try inserting or searching one.")
        ss.bst_view = {"path": [], "focus": None, "order": None}
        ss.bst_steps = []
        ss.bst_trav = None
    tree: CategoryBST = ss.bst

    st.subheader("Module 1 - Product Category Binary Search Tree")
    intro_cards(
        "A shop with hundreds of product categories must let customers browse and find a "
        "category quickly. Scanning an unordered list gets slower as the catalogue grows. "
        "A <b>Binary Search Tree (BST)</b> keeps categories in alphabetical order and lets "
        "us ignore about half of the remaining categories with every comparison.",
        "Each <b>node</b> stores a category name plus a <code>left</code> and a <code>right</code> "
        "child. Names that come <i>before</i> go left, names that come <i>after</i> go right. "
        "<b>Search</b> and <b>insert</b> start at the root and follow one branch at a time. "
        "<b>Traversals</b> visit every node: inorder (left, node, right) gives alphabetical order, "
        "preorder (node, left, right) shows parents first, postorder (left, right, node) shows "
        "children first.",
    )

    st.markdown("#### Try it")
    with st.form("bst_form"):
        name = st.text_input("Category name", placeholder="e.g. Stationery", key="bst_name")
        col_a, col_b, _ = st.columns([1, 1, 5])
        do_insert = col_a.form_submit_button("Insert", type="primary")
        do_search = col_b.form_submit_button("Search")

    if do_insert or do_search:
        cleaned = clean_name(name)
        if not cleaned:
            ss.bst_msg = ("warning", "Please type a category name first.")
        elif do_insert:
            inserted, path, steps = tree.insert(cleaned)
            ss.bst_steps = steps
            ss.bst_view = {"path": path, "focus": path[-1] if inserted else None, "order": None}
            ss.bst_msg = (("success", f"Inserted '{cleaned}'. The tree now has {tree.size} categories.")
                          if inserted else
                          ("warning", f"'{path[-1]}' already exists, so nothing was inserted."))
        else:
            found, path, steps = tree.search(cleaned)
            ss.bst_steps = steps
            ss.bst_view = {"path": path, "focus": path[-1] if found else None, "order": None}
            ss.bst_msg = (("success", f"Found '{path[-1]}' after {len(path)} comparison(s).")
                          if found else
                          ("error", f"'{cleaned}' does not exist in the catalogue "
                                    f"(checked {len(path)} node(s))."))
        ss.bst_trav = None

    st.markdown("**Traversals**")
    t1, t2, t3, t4 = st.columns(4)
    chosen = None
    if t1.button("Inorder"):
        chosen = "Inorder"
    if t2.button("Preorder"):
        chosen = "Preorder"
    if t3.button("Postorder"):
        chosen = "Postorder"
    if t4.button("Reset sample tree"):
        for key in ("bst", "bst_msg", "bst_view", "bst_steps", "bst_trav"):
            ss.pop(key, None)
        st.rerun()
    if chosen:
        sequence = {"Inorder": tree.inorder, "Preorder": tree.preorder,
                    "Postorder": tree.postorder}[chosen]()
        ss.bst_trav = (chosen, sequence)
        ss.bst_view = {"path": [], "focus": None,
                       "order": {value: i + 1 for i, value in enumerate(sequence)}}
        ss.bst_msg = ("info", f"{chosen} traversal visited {len(sequence)} categories "
                              "(blue numbers on the tree show the visit order).")

    show_message(*ss.bst_msg)

    m1, m2, m3 = st.columns(3)
    m1.metric("Nodes in the tree", tree.size)
    m2.metric("Tree height (levels)", tree.height())
    m3.metric("Root category", tree.root.value if tree.root else "-")

    show_figure(draw_bst(tree, ss.bst_view))
    st.caption("Amber = nodes visited by the last operation. Green = the node that was found or inserted. "
               "Blue numbers = traversal order.")

    if ss.bst_trav:
        traversal_name, sequence = ss.bst_trav
        meaning = {"Inorder": "Left, Node, Right - gives alphabetical order.",
                   "Preorder": "Node, Left, Right - each parent before its children.",
                   "Postorder": "Left, Right, Node - each parent after its children."}[traversal_name]
        st.markdown(f"**{traversal_name} result** ({meaning})")
        st.markdown(" → ".join(f"`{value}`" for value in sequence))

    st.markdown("#### Step-by-step dry run of the last insert / search")
    if ss.bst_steps:
        st.dataframe(pd.DataFrame(ss.bst_steps), hide_index=True)
    else:
        st.caption("Insert or search a category above and every comparison will be listed here.")

    closing_cards(
        "<ul><li><b>Search / insert:</b> O(h), where h is the tree height</li>"
        "<li><b>Balanced tree:</b> h ≈ log₂ n, so 1,000,000 categories need only about 20 comparisons</li>"
        "<li><b>Worst case:</b> O(n) if names are inserted already sorted (the tree becomes a chain). "
        "Try inserting Aaa, Aab, Aac... to watch it happen</li>"
        "<li><b>Traversals:</b> O(n) time</li><li><b>Space:</b> O(n) for the nodes</li></ul>",
        "Category menus and sidebars, autocomplete suggestions, ordered price or date filters, and "
        "database indexes (B-trees are a close cousin of the BST).",
    )


# ======================================================================
# MODULE 2 - SKU Hash Table
# ======================================================================
def reset_hash_state() -> None:
    ss = st.session_state
    table = SKUHashTable(SMALL_TABLE_SIZE)
    mapping = {}
    for sku, product in SAMPLE_PRODUCTS:
        table.insert(sku, product)
        mapping[sku] = product
    ss.hash_table, ss.hash_dict = table, mapping
    ss.hash_msg, ss.hash_trace = None, None


def bucket_diagram(table: SKUHashTable) -> str:
    lines = []
    for index, bucket in enumerate(table.buckets):
        if bucket:
            chain = " -> ".join(f"[{sku} | {product}]" for sku, product in bucket)
            note = "    <-- collision (chained)" if len(bucket) > 1 else ""
            lines.append(f"Bucket {index}: {chain} -> None{note}")
        else:
            lines.append(f"Bucket {index}: None")
    return "\n".join(lines)


def render_hash() -> None:
    ss = st.session_state
    if "hash_table" not in ss:
        reset_hash_state()
    table: SKUHashTable = ss.hash_table

    st.subheader("Module 2 - SKU Hash Table")
    intro_cards(
        "Every product has a unique <b>SKU</b> code. At checkout the platform must fetch a product from "
        "millions of SKUs instantly. Scanning a list one by one would be far too slow.",
        "A <b>hash function</b> turns the SKU text into a bucket number, so we jump straight to one small "
        "bucket instead of searching everything. When two different SKUs land in the same bucket "
        "(a <b>collision</b>), they are <b>chained</b> in a list inside that bucket and we compare only "
        "within that short list. Python's built-in <code>dict</code> uses the same idea, written in "
        "optimised C.",
    )

    st.markdown("#### Part A - A small table you can see (8 buckets, chaining)")
    st.caption(f"Hash function: start with 0, then for every character do "
               f"hash = (hash × 31 + ASCII code) mod {SMALL_TABLE_SIZE}. "
               "With more SKUs than buckets, a collision is guaranteed (pigeonhole principle).")

    st.code(bucket_diagram(table), language=None)
    collision = table.find_collision()
    if collision:
        index, sku_a, sku_b = collision
        st.warning(f"Collision demo: **{sku_a}** and **{sku_b}** are different SKUs, but both hash to "
                   f"bucket **{index}**. Chaining keeps both safely in a list inside that bucket.")
    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs stored", table.count)
    c2.metric("Load factor (SKUs per bucket)", f"{table.load_factor():.2f}")
    c3.metric("Longest chain", table.longest_chain())

    left, right = st.columns(2)
    with left:
        with st.form("hash_insert"):
            st.markdown("**Insert a SKU**")
            new_sku = st.text_input("SKU", placeholder="SKU-2001")
            new_product = st.text_input("Product name", placeholder="Gaming Headset")
            add = st.form_submit_button("Insert SKU", type="primary")
    with right:
        with st.form("hash_search"):
            st.markdown("**Search for a SKU**")
            look_sku = st.text_input("SKU to look up", value="SKU-1003")
            go = st.form_submit_button("Search SKU", type="primary")
        if st.button("Reset sample table"):
            reset_hash_state()
            st.rerun()

    if add:
        sku, product = new_sku.strip().upper(), " ".join(new_product.split())
        if not sku or not product:
            ss.hash_msg = ("warning", "Please enter both a SKU and a product name.")
        else:
            info = table.insert(sku, product)
            ss.hash_dict[sku] = product
            if info["action"] == "updated":
                ss.hash_msg = ("info", f"{sku} already existed, so its product name was updated "
                                       f"(bucket {info['index']}).")
            elif info["collision"]:
                ss.hash_msg = ("warning", f"Inserted {sku} into bucket {info['index']}. Collision! "
                                          f"The bucket now holds a chain of {info['chain_length']} SKUs.")
            else:
                ss.hash_msg = ("success", f"Inserted {sku} into empty bucket {info['index']}.")
            ss.hash_trace = None
    if ss.hash_msg:
        show_message(*ss.hash_msg)

    if go:
        sku = look_sku.strip().upper()
        if sku:
            ss.hash_trace = table.search(sku)
            ss.hash_trace["dict_result"] = ss.hash_dict.get(sku)   # same lookup with built-in dict
        else:
            ss.hash_trace = None
            st.warning("Please enter a SKU to look up.")

    trace = ss.hash_trace
    if trace:
        st.markdown(f"##### Lookup result for {trace['sku']}")
        if trace["found"]:
            st.success(f"Found: **{trace['product']}** in bucket {trace['index']} "
                       f"after {len(trace['steps'])} comparison(s).")
        else:
            st.error(f"SKU {trace['sku']} was not found (bucket {trace['index']} "
                     f"holds {trace['chain_length']} SKU(s), none matched).")
        st.caption("Same lookup with Python's built-in dictionary: "
                   + (f"found '{trace['dict_result']}'" if trace["dict_result"] is not None else "not found")
                   + (" - matches the custom table." if trace["dict_result"] == trace["product"]
                      else " - DIFFERENT from the custom table!"))
        d1, d2 = st.columns(2)
        with d1:
            st.markdown("**Dry run, part 1: compute the bucket**")
            st.dataframe(pd.DataFrame(hash_steps(trace["sku"], SMALL_TABLE_SIZE)), hide_index=True)
            st.caption(f"Final hash = {hash_function(trace['sku'], SMALL_TABLE_SIZE)} → bucket {trace['index']}")
        with d2:
            st.markdown("**Dry run, part 2: walk the chain**")
            if trace["steps"]:
                st.dataframe(pd.DataFrame(trace["steps"]), hide_index=True)
            else:
                st.caption("The bucket is empty, so there is nothing to compare.")

    st.markdown("#### Part B - Benchmark: custom hash table vs Python dict")
    st.caption("Part A used 8 buckets so you can see the chains. For 100,000 SKUs a table needs many more "
               "buckets, otherwise chains become long and lookups slow down. Try the smallest option to see that.")
    b1, b2 = st.columns(2)
    lookups = b1.select_slider("Number of lookups", options=[10_000, 50_000, 100_000, 200_000],
                               value=100_000, format_func=lambda v: f"{v:,}")
    buckets = b2.select_slider("Buckets in the benchmark table", options=[1_009, 16_381, 131_071],
                               value=131_071, format_func=lambda v: f"{v:,}")
    if st.button("Run benchmark (100,000 SKUs)", type="primary"):
        with st.spinner("Generating 100,000 SKUs and timing the lookups..."):
            ss.bench = run_benchmark(num_skus=100_000, num_lookups=lookups, table_size=buckets)

    bench = ss.get("bench")
    if bench:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Number of SKUs", f"{bench['num_skus']:,}")
        m2.metric("Lookup operations", f"{bench['num_lookups']:,}")
        m3.metric("Custom hash table", f"{bench['custom_seconds'] * 1000:,.1f} ms")
        m4.metric("Built-in dict", f"{bench['dict_seconds'] * 1000:,.1f} ms")

        if bench["ratio"] >= 1:
            st.info(f"Relative performance: Python's dict was about **{bench['ratio']:.1f}× faster** than our "
                    "pure-Python hash table. Both are O(1) on average, but dict is written in C.")
        else:
            st.info(f"Relative performance: our hash table was about **{1 / bench['ratio']:.1f}× faster** "
                    "in this run.")

        fig, ax = new_figure(7, 2.2)
        names, times = ["Built-in dict", "Custom hash table"], [bench["dict_seconds"] * 1000,
                                                              bench["custom_seconds"] * 1000]
        bars = ax.barh(names, times, color=[GREEN, BLUE], height=0.5)
        for bar, value in zip(bars, times):
            ax.text(value, bar.get_y() + bar.get_height() / 2, f"  {value:,.1f} ms",
                    va="center", color=INK, fontsize=9)
        ax.set_xlim(0, max(times) * 1.25)
        ax.set_xlabel("Total lookup time (lower is better)", color=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
        fig.tight_layout()
        show_figure(fig)

        st.caption(f"Custom table: {bench['table_size']:,} buckets, load factor {bench['load_factor']:.2f}, "
                   f"longest chain {bench['longest_chain']}. Fastest of {bench['repeats']} runs. "
                   f"Both structures returned identical results: {'yes' if bench['results_match'] else 'NO'}. "
                   "Actual timings depend on your machine, Python version and what else is running; "
                   "the ratio matters more than the absolute numbers.")

    closing_cards(
        "<ul><li><b>Insert / search:</b> O(1) on average, O(n) in the worst case "
        "(all SKUs in one bucket)</li>"
        "<li><b>Load factor</b> = SKUs ÷ buckets. Keep it small (about 1 or less) so chains stay short</li>"
        "<li><b>Space:</b> O(n + number of buckets)</li></ul>",
        "Product and SKU lookup, shopping-cart sessions, caches, spotting duplicate orders, "
        "and counting how many times each product was viewed.",
    )


# ======================================================================
# MODULE 3 - Microservice dependency graph
# ======================================================================
def draw_dependency_graph(graph: dict, positions: dict, bad_edges: set, order: list):
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    width = max(7.0, 2.9 * (max(xs) - min(xs) + 1))
    height = max(3.4, 1.5 * (max(ys) - min(ys) + 1) + 0.8)
    fig, ax = new_figure(width, height)
    ax.axis("off")

    rank = {name: i + 1 for i, name in enumerate(order or [])}
    boxes = {}
    for name, (x, y) in positions.items():
        label = textwrap.fill(name, 12) + (f"\n(deploy #{rank[name]})" if name in rank else "")
        boxes[name] = ax.text(x, y, label, ha="center", va="center", fontsize=9, color=INK, zorder=3,
                              bbox=dict(boxstyle="round,pad=0.5", fc=WHITE, ec=BLUE, lw=1.6))

    for source in graph:
        for target in graph[source]:
            bad = (source, target) in bad_edges
            if source == target:   # a service that depends on itself
                x, y = positions[source]
                ax.text(x, y + 0.45, "depends on itself", ha="center", color=RED, fontsize=8, zorder=5)
                continue
            curved = bad or target in graph.get(target, []) and source in graph.get(target, [])
            ax.annotate("", xy=positions[target], xytext=positions[source], zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=RED if bad else MUTED,
                                        lw=2.2 if bad else 1.6, shrinkA=0, shrinkB=0,
                                        patchA=boxes[source].get_bbox_patch(),
                                        patchB=boxes[target].get_bbox_patch(),
                                        connectionstyle="arc3,rad=0.15" if curved else "arc3,rad=0"))

    ax.set_xlim(min(xs) - 0.8, max(xs) + 0.8)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    fig.tight_layout()
    return fig


def render_graph() -> None:
    ss = st.session_state
    ss.setdefault("dep_run", None)
    ss.setdefault("dep_version", 0)

    st.subheader("Module 3 - Microservice Dependency Graph")
    intro_cards(
        "The Cart service needs Pricing and Inventory, Pricing needs the Product Catalogue, and Inventory "
        "needs the Warehouse. If Cart is deployed before the services it depends on, the release breaks. "
        "We need an order where every service comes <i>after</i> everything it depends on, and a clear "
        "error when no such order exists.",
        "Services are <b>nodes</b>; dependencies are <b>directed edges</b> (an arrow A → B means "
        "<i>deploy A before B</i>). <b>Topological sort</b> lines the nodes up so every arrow points forward. "
        "<b>BFS / Kahn's algorithm</b> repeatedly deploys services that have no unmet dependencies. "
        "<b>DFS</b> goes deep, finishes a service once everything waiting for it is done, and the reversed "
        "finish order is the answer. With a <b>circular dependency</b> no order exists: Kahn's queue runs "
        "dry early, and DFS meets a service that is still in progress.",
    )

    st.markdown("#### Choose or edit the dependencies")
    scenario = st.radio("Sample graph", ["Valid sample (from the brief)", "Cyclic demo (circular dependency)"],
                        horizontal=True)
    base = SAMPLE_DEPENDENCIES if scenario.startswith("Valid") else CYCLIC_DEPENDENCIES
    st.caption("Each row means: *Service* depends on *Depends on* (so 'Depends on' is deployed first). "
               "Edit a cell, add a row with the + at the bottom of the table, or delete a row.")
    editor_key = f"dep_editor_{scenario[:5]}_{ss.dep_version}"
    edited = st.data_editor(
        pd.DataFrame(base, columns=["Service", "Depends on"]), num_rows="dynamic", key=editor_key,
        column_config={
            "Service": st.column_config.SelectboxColumn("Service", options=ALL_SERVICES, required=True),
            "Depends on": st.column_config.SelectboxColumn("Depends on", options=ALL_SERVICES, required=True),
        })
    if st.button("Restore this sample"):
        ss.dep_version += 1
        st.rerun()

    dependencies = [(a, b) for a, b in zip(edited["Service"], edited["Depends on"])
                    if pd.notna(a) and pd.notna(b)]
    graph = build_graph(DEFAULT_SERVICES, dependencies)
    cycle = find_cycle(graph)

    st.markdown("#### Run an algorithm")
    b1, b2, b3 = st.columns(3)
    if b1.button("Run BFS ordering (Kahn)", type="primary"):
        ss.dep_run = "bfs"
    if b2.button("Run DFS ordering"):
        ss.dep_run = "dfs"
    if b3.button("Detect circular dependencies"):
        ss.dep_run = "cycle"

    run = ss.dep_run
    result = kahn_topological_sort(graph) if run == "bfs" else dfs_topological_sort(graph) if run == "dfs" else None
    order = result["order"] if result and not result["has_cycle"] else []

    show_figure(draw_dependency_graph(graph, compute_layout(graph, cycle is not None),
                                      cycle_edges(cycle), order))
    st.caption("Arrow A → B: deploy A before B. Red arrows are part of a circular dependency.")

    if run in ("bfs", "dfs"):
        title = "BFS (Kahn's algorithm)" if run == "bfs" else "DFS"
        st.markdown(f"##### {title} result")
        if result["has_cycle"]:
            if run == "bfs":
                st.error("No valid deployment order exists: there is a circular dependency. "
                         f"These services can never be deployed: {', '.join(result['stuck'])}.")
                if result["order"]:
                    st.caption("Deployed before getting stuck: " + " → ".join(result["order"]))
            else:
                st.error("No valid deployment order exists: circular dependency "
                         + " → ".join(result["cycle"]) + ".")
        else:
            st.success("Valid deployment order: " + " → ".join(f"{i + 1}. {n}" for i, n in enumerate(order)))
            st.caption("Order check: every service comes after all of its dependencies - "
                       + ("verified." if verify_order(graph, order) else "FAILED."))
            st.caption("BFS and DFS can produce different orders. Both are valid as long as every "
                       "arrow points forward.")
        st.markdown("**Step-by-step dry run**")
        if result["steps"]:
            st.dataframe(pd.DataFrame(result["steps"]), hide_index=True)
        else:
            st.caption("No steps were needed.")
    elif run == "cycle":
        st.markdown("##### Cycle detection result")
        if cycle:
            st.error("Circular dependency detected: " + " → ".join(cycle)
                     + "  (read each arrow as 'must be deployed before')")
        else:
            st.success("No circular dependencies. A valid deployment order exists.")
        kahn = kahn_topological_sort(graph)
        st.caption("Cross-check with Kahn's algorithm: "
                   + (f"stuck services that never reach zero dependencies: {', '.join(kahn['stuck'])}."
                      if kahn["has_cycle"] else "every service could be deployed, so there is no cycle."))
        st.markdown("**DFS dry run (colour method)**")
        st.dataframe(pd.DataFrame(dfs_topological_sort(graph)["steps"]), hide_index=True)
    else:
        st.caption("Press one of the buttons above to run an algorithm.")

    closing_cards(
        "<ul><li><b>BFS (Kahn) and DFS:</b> O(V + E) time, where V = services and E = dependencies</li>"
        "<li><b>Space:</b> O(V + E) for the graph, plus O(V) for the queue / stack / colours</li>"
        "<li>Cycle detection comes free: Kahn leaves services undeployed, DFS finds a back edge</li></ul>",
        "CI/CD release pipelines, build tools (make, Gradle), package installers (pip, npm), course "
        "prerequisites, and any task scheduler where some jobs must finish first.",
    )


# ======================================================================
# MODULE 4 - Traffic analysis (sliding window)
# ======================================================================
def draw_traffic(values: list, result: dict, window: int):
    start, end = result["best_start"], result["best_end"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6.2), gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(BG)
    hours = list(range(len(values)))

    colors = [AMBER if start <= h <= end else SOFT_BLUE for h in hours]
    ax1.bar(hours, values, color=colors, width=0.75)
    ax1.axvspan(start - 0.5, end + 0.5, color=AMBER, alpha=0.12)
    ax1.set_title(f"Hourly visitors - busiest {window}-hour window in amber "
                  f"({result['best_sum']:,} visits)", loc="left", fontsize=10, color=INK)
    ax1.set_ylabel("Visitors", color=MUTED)

    sums = result["window_sums"]
    ax2.plot(range(len(sums)), sums, color=BLUE, lw=2)
    ax2.scatter([start], [result["best_sum"]], s=70, color=AMBER, zorder=3, edgecolor=WHITE, linewidth=2)
    ax2.annotate(f"max {result['best_sum']:,}", (start, result["best_sum"]), textcoords="offset points",
                 xytext=(8, -14), fontsize=9, color=INK)
    ax2.set_title(f"Total of each {window}-hour window, by its start hour", loc="left", fontsize=10, color=INK)
    ax2.set_ylabel("Window total", color=MUTED)
    ax2.set_xlabel("Hour of day (start hour of the window)", color=MUTED)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.set_xticks(range(0, len(values), 2))
        ax.set_xlim(-0.7, len(values) - 0.3)
        ax.grid(axis="y", color=GRID, lw=0.7, alpha=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
    fig.tight_layout()
    return fig


def render_traffic() -> None:
    ss = st.session_state
    if "traffic_values" not in ss:
        ss.traffic_values = list(DEFAULT_TRAFFIC)
        ss.traffic_version = 0

    st.subheader("Module 4 - Traffic Analysis Using a Sliding Window")
    intro_cards(
        "To size servers for peak traffic, the team wants the <b>busiest continuous 3-hour period</b> of the "
        "day. Looking at the single busiest hour hides sustained load, and re-adding every 3-hour block "
        "from scratch wastes work.",
        "A <b>fixed-size sliding window</b> adds up the first <i>k</i> hours once. To move one hour forward "
        "it <b>subtracts the hour that leaves</b> and <b>adds the hour that enters</b>, which is just two "
        "operations. While sliding, it remembers the biggest total and where that window started.",
    )

    st.markdown("#### Try it")
    window = st.slider("Window size (hours)", min_value=1, max_value=24, value=3)
    r1, r2, _ = st.columns([1, 1, 4])
    if r1.button("Reset to sample data"):
        ss.traffic_values = list(DEFAULT_TRAFFIC)
        ss.traffic_version += 1
    if r2.button("Randomize traffic"):
        ss.traffic_values = generate_random_traffic()
        ss.traffic_version += 1

    table_col, result_col = st.columns([2, 5])
    with table_col:
        frame = pd.DataFrame({"Hour": [f"{h:02d}:00" for h in range(24)], "Visitors": ss.traffic_values})
        edited = st.data_editor(
            frame, hide_index=True, disabled=["Hour"], height=460, key=f"traffic_editor_{ss.traffic_version}",
            column_config={"Visitors": st.column_config.NumberColumn("Visitors", min_value=0, step=1, format="%d")})
    values = [int(v) if pd.notna(v) else 0 for v in edited["Visitors"]]

    result = sliding_window_max(values, window)
    naive = naive_window_max(values, window)

    with result_col:
        st.success(f"Busiest {window}-hour window: **{format_hour_range(result['best_start'], window)}** "
                   f"with **{result['best_sum']:,}** visits.")
        k1, k2, k3 = st.columns(3)
        k1.metric("Starting hour", f"{result['best_start']:02d}:00")
        k2.metric("Ending hour", f"{result['best_end']:02d}:59")
        k3.metric("Maximum traffic total", f"{result['best_sum']:,}")
        show_figure(draw_traffic(values, result, window))

    st.markdown("#### Step-by-step dry run")
    st.caption("Each row slides the window one hour: new total = old total − removed + added.")
    rows = pd.DataFrame(result["steps"])
    best = result["best_start"]
    st.dataframe(rows.style.apply(
        lambda row: [f"background-color: {SOFT_AMBER}" if row["Start hour"] == best else "" for _ in row], axis=1),
        hide_index=True)

    st.markdown("#### Sliding window vs naive approach")
    agree = (naive["best_start"] == result["best_start"] and naive["best_sum"] == result["best_sum"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Sliding window operations", f"{result['operations']:,}")
    c2.metric("Naive operations (re-add every window)", f"{naive['operations']:,}")
    c3.metric("Both methods agree", "Yes" if agree else "NO")
    st.caption("Sliding window: O(n) time. Naive: O(n × k) time. With only 24 values the difference is tiny, "
               "but for a month of per-minute data (43,200 values) and a 3-hour window (k = 180) the naive "
               "method needs about 7.7 million additions versus about 86 thousand operations.")

    closing_cards(
        "<ul><li><b>Sliding window:</b> O(n) time, O(1) extra space (n = number of hours)</li>"
        "<li><b>Naive approach:</b> O(n × k) time, because each window is added from scratch</li>"
        "<li>The window size k does not change the sliding window's running time</li></ul>",
        "Peak-load capacity planning, rate limiting, moving averages on dashboards, spotting bursts of "
        "fraud or errors, and finding the best sales streak.",
    )


# ======================================================================
# MODULE 5 - Gift card pair finder (two pointers)
# ======================================================================
def draw_pointers(values: list, step: dict, target: int):
    left, right = step["Left index"], step["Right index"]
    matched = step["Sum"] == target
    count = len(values)
    fig, ax = new_figure(max(6.0, count * 0.8), 2.6)
    ax.axis("off")

    for index, value in enumerate(values):
        if index == left or index == right:
            face = GREEN if matched else (BLUE if index == left else AMBER)
            text_color, edge = WHITE if (matched or index == left) else INK, face
        elif left < index < right:
            face, text_color, edge = WHITE, INK, BLUE
        else:  # already ruled out by the pointers
            face, text_color, edge = "#E2E8F0", MUTED, GRID
        ax.add_patch(FancyBboxPatch((index - 0.4, -0.35), 0.8, 0.7, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=face, ec=edge, lw=1.6))
        ax.text(index, 0, str(value), ha="center", va="center", fontsize=10, color=text_color, fontweight="bold")
        ax.text(index, -0.62, str(index), ha="center", va="center", fontsize=7.5, color=MUTED)

    ax.text(left, -1.05, "▲\nLEFT", ha="center", va="top", fontsize=8.5, color=BLUE, fontweight="bold")
    ax.text(right, 0.62, "RIGHT\n▼", ha="center", va="bottom", fontsize=8.5, color="#B45309", fontweight="bold")
    ax.set_xlim(-0.7, count - 0.3)
    ax.set_ylim(-1.9, 1.5)
    fig.tight_layout()
    return fig


def render_gift() -> None:
    ss = st.session_state
    if "gift" not in ss:
        ss.gift = {"text": ", ".join(str(v) for v in DEFAULT_ORDER_VALUES), "target": DEFAULT_TARGET}

    st.subheader("Module 5 - Gift Card Pair Finder")
    intro_cards(
        "A customer holds a gift card worth a certain amount. The platform wants to suggest <b>two order "
        "values that add up exactly to the gift-card value</b>, so no balance is left over.",
        "With values <b>sorted</b>, put a <b>left</b> pointer on the smallest and a <b>right</b> pointer on "
        "the largest. If the sum is <b>too small</b>, move left to a bigger value. If it is <b>too big</b>, "
        "move right to a smaller value. If it <b>matches</b>, save the pair and move both inward. Each move "
        "rules out a value that can never be in a pair, so one pass is enough.",
    )

    st.markdown("#### Try it")
    with st.form("gift_form"):
        text = st.text_input("Order values (whole numbers, comma separated)", value=ss.gift["text"])
        target_in = st.number_input("Gift-card value", min_value=0, step=1, value=int(ss.gift["target"]))
        submitted = st.form_submit_button("Find pairs", type="primary")
    if submitted:
        ss.gift = {"text": text, "target": int(target_in)}

    try:
        entered = parse_int_list(ss.gift["text"])
    except ValueError as error:
        st.error(str(error))
        return
    target = ss.gift["target"]

    if is_sorted(entered):
        values = entered
    else:
        values = []
        for number in entered:          # insertion sort, to keep sorting beginner friendly
            position = len(values)
            while position > 0 and values[position - 1] > number:
                position -= 1
            values.insert(position, number)
        st.info("Your values were not in order, so they were sorted first (the two-pointer method needs "
                "sorted data). Sorted list: " + ", ".join(str(v) for v in values))

    outcome = two_pointer_pairs(values, target)
    pairs, steps = outcome["pairs"], outcome["steps"]

    st.markdown("#### Result")
    if pairs:
        st.success(f"Found {len(pairs)} pair(s) that add up to {target}: "
                   + ",  ".join(f"{a} + {b}" for a, b in pairs))
    else:
        st.warning(f"No two order values add up to {target}.")

    st.markdown("#### Pointer movements")
    if not steps:
        st.caption("Enter at least two order values to see the pointers move.")
    elif len(values) > 30:
        st.caption("The picture is shown for up to 30 values. The full dry-run table is below.")
    else:
        if len(steps) > 1:
            number = st.slider("Step", 1, len(steps), 1, key=f"gift_step_{len(steps)}_{target}_{len(values)}")
        else:
            number = 1
        step = steps[number - 1]
        show_figure(draw_pointers(values, step, target))
        st.caption(f"Step {number}: {step['Left value']} + {step['Right value']} = {step['Sum']}. {step['Action']}. "
                   "Grey boxes are values the pointers have already ruled out. Green = matching pair.")

    st.markdown("#### Step-by-step dry run")
    if steps:
        st.dataframe(pd.DataFrame(steps), hide_index=True)

    brute = brute_force_pairs(values, target)
    c1, c2, c3 = st.columns(3)
    c1.metric("Two-pointer comparisons", outcome["comparisons"])
    c2.metric("Brute-force comparisons", brute["comparisons"])
    c3.metric("Same pairs found", "Yes" if brute["pairs"] == pairs else "NO")

    with st.expander("Why does the algorithm need sorted input?", expanded=True):
        demo = unsorted_demo()
        st.markdown(
            "Moving a pointer is only safe when we know how the sum will change. In **sorted** data, moving "
            "the left pointer right can only make the sum bigger, and moving the right pointer left can only "
            "make it smaller. In **unsorted** data that promise is gone, so the algorithm can throw away a "
            "value that was actually needed.")
        st.markdown(f"Example with values `{demo['values']}` and target `{demo['target']}` (not sorted):")
        st.markdown(f"- Two pointers on the unsorted list found: `{demo['two_pointer_result']}`\n"
                    f"- The correct answer is: `{demo['correct_result']}`\n"
                    f"- Missed pairs: `{demo['missed']}`")

    closing_cards(
        "<ul><li><b>Two pointers:</b> O(n) time and O(1) extra space once the data is sorted</li>"
        "<li><b>Sorting first</b> (if needed): O(n log n)</li>"
        "<li><b>Brute force:</b> O(n²), because it checks every pair</li></ul>",
        "Bundling two products to use a gift card or voucher exactly, matching a budget, finding "
        "complementary items, reconciling payments, and the classic 'two-sum' problem in interviews.",
    )


# ======================================================================
# Decision table
# ======================================================================
def render_decision_table() -> None:
    st.subheader("Decision Table")
    st.markdown("Which data structure or algorithm fits which business problem, and why.")
    st.markdown(
        "| Business Problem | Data Structure / Algorithm | Reason |\n"
        "|---|---|---|\n"
        "| Product category browsing | Binary Search Tree | Ordered insertion and searching |\n"
        "| SKU lookup | Hash Table with Chaining | Average constant-time key lookup |\n"
        "| Built-in SKU lookup | Python Dictionary | Optimized hash-map implementation |\n"
        "| Microservice deployment | Graph + BFS / DFS | Dependency ordering and cycle detection |\n"
        "| Peak traffic analysis | Sliding Window | Efficient continuous time-window calculations |\n"
        "| Gift-card pair matching | Two Pointers | Efficient pair search in sorted data |\n")

    st.markdown("#### Where each one lives in the code")
    st.markdown(
        "| Module | Algorithm file | Time complexity |\n"
        "|---|---|---|\n"
        "| Product Category BST | `bst_catalogue.py` | O(h) search / insert, O(log n) when balanced |\n"
        "| SKU Hash Table | `sku_hash_table.py` | O(1) average lookup |\n"
        "| Microservice Dependency Graph | `service_dependencies.py` | O(V + E) |\n"
        "| Traffic Analysis | `traffic_analysis.py` | O(n) |\n"
        "| Gift Card Pair Finder | `gift_card_pairs.py` | O(n) on sorted data |\n")


# ======================================================================
# Main
# ======================================================================
def main() -> None:
    st.set_page_config(page_title="E-Commerce Catalogue and Traffic Analysis", page_icon="🛒", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    st.sidebar.markdown("### DSA Prototype")
    page = st.sidebar.radio("Modules", PAGES, label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption("Educational prototype. All five algorithms are written from scratch in plain Python.")

    st.title("E-Commerce Catalogue and Traffic Analysis")
    st.markdown(
        '<div class="hero"><b>Business scenario:</b> an e-commerce platform wants <b>faster category '
        'browsing</b>, <b>instant SKU look-ups</b> and a way to <b>size infrastructure for peak traffic</b>. '
        "Pick a module in the sidebar to see which data structure or algorithm solves each problem.</div>",
        unsafe_allow_html=True)

    renderers = {PAGES[0]: render_bst, PAGES[1]: render_hash, PAGES[2]: render_graph,
                 PAGES[3]: render_traffic, PAGES[4]: render_gift, PAGES[5]: render_decision_table}
    renderers[page]()


if __name__ == "__main__":
    main()
