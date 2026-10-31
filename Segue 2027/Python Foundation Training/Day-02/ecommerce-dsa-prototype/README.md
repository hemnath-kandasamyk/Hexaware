# E-Commerce Catalogue and Traffic Analysis — DSA Interactive Prototype

A beginner-friendly Streamlit dashboard that shows how five Data Structures and
Algorithms (DSA) concepts solve real e-commerce problems:

> "An e-commerce platform wants faster category browsing, instant SKU look-ups
> and a way to size infrastructure for peak traffic."

This is an educational prototype, not a production application. Every algorithm
is written by hand in plain Python and kept separate from the user interface.

## Modules

| # | Module | Business problem | DSA concept | File |
|---|--------|------------------|-------------|------|
| 1 | Product Category BST | Fast category browsing and search | Binary Search Tree | `bst_catalogue.py` |
| 2 | SKU Hash Table | Instant SKU lookups | Hash table with chaining (vs Python `dict`) | `sku_hash_table.py` |
| 3 | Microservice Dependency Graph | Deploy services in the right order | Directed graph, BFS (Kahn) and DFS topological sort, cycle detection | `service_dependencies.py` |
| 4 | Traffic Analysis | Busiest continuous 3-hour period | Fixed-size sliding window | `traffic_analysis.py` |
| 5 | Gift Card Pair Finder | Two order values that match a gift card | Two pointers on sorted data | `gift_card_pairs.py` |

A **Decision Table** page summarises which technique fits which problem and why.

Every module page contains: business problem, algorithm overview, interactive
inputs, algorithm execution output, a step-by-step dry run, time and space
complexity, and a real-world application.

## Project structure

```
ecommerce-dsa-prototype/
├── app.py                    # Streamlit dashboard (UI only)
├── bst_catalogue.py          # Module 1: Node + BST (insert, search, traversals)
├── sku_hash_table.py         # Module 2: hash table with chaining + benchmark
├── service_dependencies.py   # Module 3: graph, Kahn's BFS, DFS, cycle detection
├── traffic_analysis.py       # Module 4: sliding window + naive comparison
├── gift_card_pairs.py        # Module 5: two-pointer pair finder
├── requirements.txt
├── README.md
└── .streamlit/config.toml    # light theme settings
```

## Requirements

- Python 3.12 (3.10 or newer also works)
- Packages: `streamlit`, `matplotlib`, `pandas` (listed in `requirements.txt`)

## Installation

Open a terminal inside the `ecommerce-dsa-prototype` folder.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once and try again.

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

Your browser opens at <http://localhost:8501>. Use the sidebar to switch between
the five modules and the decision table. Press `Ctrl+C` in the terminal to stop.

## How to use each module

1. **Product Category BST** — type a category, press *Insert* or *Search*. Use the
   *Inorder / Preorder / Postorder* buttons to see numbered visit orders on the tree.
   Amber nodes are the path the last operation walked; green is the node found or inserted.
2. **SKU Hash Table** — see the 8 buckets and the collision demo, insert or search
   SKUs, then press *Run benchmark* to compare the custom table with Python's
   `dict` on 100,000 SKUs.
3. **Microservice Dependency Graph** — pick the valid or cyclic sample, edit the
   dependency table, then run BFS, DFS or cycle detection. A cyclic graph shows an
   error instead of a deployment order.
4. **Traffic Analysis** — edit the 24 hourly values, change the window size
   (default 3) and watch the busiest window highlighted in the chart.
5. **Gift Card Pair Finder** — enter order values and a gift-card amount, press
   *Find pairs*, then move the *Step* slider to watch the left and right pointers.

## Notes

- **Benchmark results vary.** Timings depend on your machine, Python version and what
  else is running. The ratio between the two methods matters more than the exact
  milliseconds. Python's `dict` is implemented in C, so it is normally faster than
  a hash table written in pure Python, even though both are O(1) on average.
- **Hash table size.** Part A of Module 2 uses 8 buckets so collisions are easy to see.
  The benchmark uses a larger table. Choose the smallest option there to see long
  chains slow lookups down.
- **Category names** are compared case-insensitively (`books` equals `Books`).
- **Order values** in Module 5 are whole numbers. Unsorted input is sorted for you,
  because two pointers only works on sorted data.
- The dependency table in Module 3 keeps your edits while you stay on that page.
  Switching to another page and back restores the sample.

## Decision table

| Business Problem | Data Structure / Algorithm | Reason |
|---|---|---|
| Product category browsing | Binary Search Tree | Ordered insertion and searching |
| SKU lookup | Hash Table with Chaining | Average constant-time key lookup |
| Built-in SKU lookup | Python Dictionary | Optimized hash-map implementation |
| Microservice deployment | Graph + BFS / DFS | Dependency ordering and cycle detection |
| Peak traffic analysis | Sliding Window | Efficient continuous time-window calculations |
| Gift-card pair matching | Two Pointers | Efficient pair search in sorted data |

## Troubleshooting

- `streamlit: command not found` — activate the virtual environment, or run `python -m streamlit run app.py`.
- Port already in use — run `streamlit run app.py --server.port 8502`.
- Run the command from inside the project folder so the modules can be imported.
