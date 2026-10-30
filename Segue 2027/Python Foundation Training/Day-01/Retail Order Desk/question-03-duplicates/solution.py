import csv
from datetime import datetime
from decimal import Decimal
from pathlib import Path


def find_duplicates(orders):
    seen = set()
    duplicates = []

    for order in orders:
        timestamp = datetime.strptime(
            order["createdAt"], "%Y-%m-%d %H:%M:%S"
        )

        # Ignore seconds: compare within the same calendar minute.
        minute = timestamp.strftime("%Y-%m-%d %H:%M")

        # Decimal avoids floating-point equality issues for money.
        amount = Decimal(order["amount"]).quantize(Decimal("0.01"))

        key = (order["customerId"], amount, minute)

        if key in seen:
            duplicates.append(order)
        else:
            seen.add(key)

    return duplicates


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[1] / "data" / "orders_sample.csv"

    with open(data_path, newline="", encoding="utf-8") as file:
        orders = list(csv.DictReader(file))

    for order in find_duplicates(orders):
        print("Duplicate candidate:", order["orderId"])
