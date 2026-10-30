import csv
import time
from pathlib import Path


def load_orders(path):
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def linear_search(orders, target_id):
    for order in orders:
        if order["orderId"] == target_id:
            return order
    return None


def binary_search(orders, target_id):
    left, right = 0, len(orders) - 1

    while left <= right:
        mid = (left + right) // 2
        current_id = orders[mid]["orderId"]

        if current_id == target_id:
            return orders[mid]
        if current_id < target_id:
            left = mid + 1
        else:
            right = mid - 1

    return None


def benchmark(search_fn, orders, target_id, repetitions=100):
    start = time.perf_counter()

    for _ in range(repetitions):
        search_fn(orders, target_id)

    elapsed = time.perf_counter() - start
    return elapsed / repetitions


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1] / "data" / "orders_sample.csv"
    orders = load_orders(data_path)

    # Binary search requires records sorted by orderId.
    orders.sort(key=lambda order: order["orderId"])
    target = "ORD104"

    print("Linear:", linear_search(orders, target))
    print("Binary:", binary_search(orders, target))
    print("Linear average (seconds):",
          benchmark(linear_search, orders, target))
    print("Binary average (seconds):",
          benchmark(binary_search, orders, target))
