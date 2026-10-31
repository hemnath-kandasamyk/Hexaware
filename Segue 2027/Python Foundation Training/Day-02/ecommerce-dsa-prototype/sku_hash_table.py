"""
sku_hash_table.py
=================
Module 2 - SKU lookup with a hash table (separate chaining), written from scratch.

Business problem
    Checkout and product pages must fetch a product from its unique SKU code
    instantly, even with huge catalogues.

How a hash table helps
    A hash function turns the SKU text into a bucket number, so we jump straight
    to one small bucket instead of scanning every product. If two different SKUs
    land in the same bucket (a "collision") they are chained in a list inside
    that bucket.

Python's built-in dict is used only for the comparison. The custom table does
not use dict, set, or hash().
"""

import random
import time

SMALL_TABLE_SIZE = 8  # small on purpose, so collisions are easy to see

SAMPLE_PRODUCTS = [
    ("SKU-1001", "Wireless Mouse"),
    ("SKU-1002", "Mechanical Keyboard"),
    ("SKU-1003", "USB-C Charger"),
    ("SKU-1004", "Running Shoes"),
    ("SKU-1005", "Cotton T-Shirt"),
    ("SKU-1006", "Water Bottle"),
    ("SKU-1007", "Yoga Mat"),
    ("SKU-1008", "Desk Lamp"),
    ("SKU-1009", "Backpack"),
    ("SKU-1010", "Bluetooth Speaker"),
]

_PRODUCT_TYPES = ["Mouse", "Keyboard", "Charger", "Shoes", "T-Shirt",
                  "Bottle", "Yoga Mat", "Lamp", "Backpack", "Speaker"]


def hash_function(key: str, table_size: int) -> int:
    """
    Turn a SKU string into a bucket number between 0 and table_size - 1.

    For every character: hash = (hash * 31 + ASCII code of the character) mod table_size
    Using 31 spreads similar SKUs (SKU-1001, SKU-1002 ...) across different buckets.
    """
    hash_value = 0
    for character in key:
        hash_value = (hash_value * 31 + ord(character)) % table_size
    return hash_value


def hash_steps(key: str, table_size: int) -> list:
    """The same calculation as hash_function, one row per character (for the dry run)."""
    rows = []
    hash_value = 0
    for character in key:
        previous = hash_value
        hash_value = (hash_value * 31 + ord(character)) % table_size
        rows.append({
            "Character": character,
            "ASCII code": ord(character),
            "Calculation": f"({previous} x 31 + {ord(character)}) mod {table_size}",
            "Running hash": hash_value,
        })
    return rows


class SKUHashTable:
    """Fixed-size hash table. Each bucket is a list of (sku, product) pairs (a chain)."""

    def __init__(self, size: int = SMALL_TABLE_SIZE):
        if size < 1:
            raise ValueError("Table size must be at least 1.")
        self.size = size
        self.buckets = [[] for _ in range(size)]
        self.count = 0  # number of stored SKUs

    def insert(self, sku: str, product: str) -> dict:
        """Add a SKU (or update its product if the SKU already exists)."""
        index = hash_function(sku, self.size)
        bucket = self.buckets[index]

        for position, (existing_sku, _) in enumerate(bucket):
            if existing_sku == sku:                    # same SKU -> just update the name
                bucket[position] = (sku, product)
                return {"index": index, "action": "updated", "collision": False,
                        "chain_length": len(bucket)}

        collision = len(bucket) > 0                    # bucket already used by another SKU
        bucket.append((sku, product))
        self.count += 1
        return {"index": index, "action": "inserted", "collision": collision,
                "chain_length": len(bucket)}

    def get(self, sku: str):
        """Fast lookup used by the benchmark. Returns the product name or None."""
        for existing_sku, product in self.buckets[hash_function(sku, self.size)]:
            if existing_sku == sku:
                return product
        return None

    def search(self, sku: str) -> dict:
        """Lookup that also records every comparison, for the step-by-step dry run."""
        index = hash_function(sku, self.size)
        bucket = self.buckets[index]
        steps = []
        found, product = False, None

        for position, (existing_sku, existing_product) in enumerate(bucket, start=1):
            matched = existing_sku == sku
            steps.append({"Step": position, "Compare with": existing_sku,
                          "Match?": "Yes" if matched else "No"})
            if matched:
                found, product = True, existing_product
                break

        return {"sku": sku, "index": index, "found": found, "product": product,
                "steps": steps, "chain_length": len(bucket)}

    # ---------------- helpers for display ----------------
    def load_factor(self) -> float:
        """Average number of SKUs per bucket."""
        return self.count / self.size

    def longest_chain(self) -> int:
        longest = 0
        for bucket in self.buckets:
            if len(bucket) > longest:
                longest = len(bucket)
        return longest

    def find_collision(self):
        """Return (bucket_index, sku_a, sku_b) for the first collision, or None."""
        for index, bucket in enumerate(self.buckets):
            if len(bucket) > 1:
                return index, bucket[0][0], bucket[1][0]
        return None


# ----------------------------------------------------------------------
# Benchmark: custom hash table vs Python's built-in dict
# ----------------------------------------------------------------------
def generate_skus(count: int = 100_000, seed: int = 42) -> list:
    """Create `count` sample (sku, product) pairs such as ('SKU-000042', 'Lamp 42')."""
    rng = random.Random(seed)
    return [(f"SKU-{number:06d}", f"{rng.choice(_PRODUCT_TYPES)} {number}")
            for number in range(count)]


def run_benchmark(num_skus: int = 100_000, num_lookups: int = 100_000,
                  table_size: int = 131_071, repeats: int = 3, seed: int = 42) -> dict:
    """
    Load the same SKUs into both structures, then time the same lookups.

    Each timing is repeated `repeats` times and the fastest run is kept, which
    reduces random noise from other programs running on the machine.
    """
    items = generate_skus(num_skus, seed)

    # Build the custom hash table
    custom = SKUHashTable(table_size)
    start = time.perf_counter()
    for sku, name in items:
        custom.insert(sku, name)
    custom_build = time.perf_counter() - start

    # Build the built-in dictionary
    builtin = {}
    start = time.perf_counter()
    for sku, name in items:
        builtin[sku] = name
    dict_build = time.perf_counter() - start

    # The SKUs we will look up (random picks, so every lookup is a hit)
    rng = random.Random(seed + 1)
    lookup_keys = [items[rng.randrange(len(items))][0] for _ in range(num_lookups)]

    # Correctness check (not timed): both structures must give identical answers
    results_match = all(custom.get(key) == builtin.get(key) for key in lookup_keys)

    def best_time(lookup) -> float:
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            for key in lookup_keys:
                lookup(key)
            best = min(best, time.perf_counter() - start)
        return best

    custom_time = best_time(custom.get)
    dict_time = best_time(builtin.get)

    return {
        "num_skus": num_skus,
        "num_lookups": num_lookups,
        "table_size": table_size,
        "repeats": repeats,
        "custom_seconds": custom_time,
        "dict_seconds": dict_time,
        "ratio": custom_time / max(dict_time, 1e-12),   # >1 means dict is faster
        "custom_build_seconds": custom_build,
        "dict_build_seconds": dict_build,
        "load_factor": custom.load_factor(),
        "longest_chain": custom.longest_chain(),
        "results_match": results_match,
    }
