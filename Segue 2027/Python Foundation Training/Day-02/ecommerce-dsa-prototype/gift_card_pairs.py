"""
gift_card_pairs.py
==================
Module 5 - Find pairs of order values that add up to a gift-card amount,
using the two-pointer technique.

Business problem
    A customer has a gift card worth a certain amount. Which two order values
    add up to exactly that amount (so no balance is left over)?

How two pointers help
    On SORTED data put one pointer at the smallest value and one at the largest.
    - sum too small -> move the left pointer right (bigger numbers)
    - sum too big   -> move the right pointer left (smaller numbers)
    - sum matches   -> record the pair and move both pointers inward
    Each move throws away a value that can never be part of a pair, so one
    pass is enough: O(n) instead of checking every pair O(n^2).
"""

DEFAULT_ORDER_VALUES = [8, 12, 15, 20, 25, 30, 35, 40, 45, 55, 60, 70]
DEFAULT_TARGET = 75


def parse_int_list(text: str) -> list:
    """'10, 20 30' -> [10, 20, 30]. Raises ValueError with a friendly message."""
    tokens = text.replace(",", " ").split()
    if not tokens:
        raise ValueError("Please enter at least one order value.")
    values = []
    for token in tokens:
        try:
            number = int(token)
        except ValueError:
            raise ValueError(f"'{token}' is not a whole number. Use values like: 10, 25, 40") from None
        if number < 0:
            raise ValueError("Order values cannot be negative.")
        values.append(number)
    return values


def is_sorted(values: list) -> bool:
    for i in range(1, len(values)):
        if values[i - 1] > values[i]:
            return False
    return True


def two_pointer_pairs(values: list, target: int, require_sorted: bool = True) -> dict:
    """
    Find all pairs of VALUES that add up to `target` (each pair listed once).

    `values` must be sorted (smallest to largest). Set require_sorted=False only
    for the "what goes wrong without sorting" demo.
    Returns {"pairs": [(a, b), ...], "steps": [...], "comparisons": int}
    """
    if require_sorted and not is_sorted(values):
        raise ValueError("Two-pointer search needs sorted values.")

    left, right = 0, len(values) - 1
    pairs, steps = [], []

    while left < right:
        a, b = values[left], values[right]
        total = a + b

        if total == target:
            action = f"Match! {a} + {b} = {target}. Save the pair, move both pointers inward"
            relation = "="
        elif total < target:
            action = "Sum is too small, so move the LEFT pointer right (to a bigger value)"
            relation = "<"
        else:
            action = "Sum is too big, so move the RIGHT pointer left (to a smaller value)"
            relation = ">"

        steps.append({
            "Step": len(steps) + 1,
            "Left index": left,
            "Right index": right,
            "Left value": a,
            "Right value": b,
            "Sum": total,
            f"Sum vs {target}": relation,
            "Action": action,
        })

        if total == target:
            pairs.append((a, b))
            left += 1
            right -= 1
            # skip repeated values so the same pair is not listed twice
            while left < right and values[left] == values[left - 1]:
                left += 1
            while left < right and values[right] == values[right + 1]:
                right -= 1
        elif total < target:
            left += 1
        else:
            right -= 1

    return {"pairs": pairs, "steps": steps, "comparisons": len(steps)}


def brute_force_pairs(values: list, target: int) -> dict:
    """The slow way, for comparison: test every pair. O(n^2). Works on unsorted data too."""
    found = set()
    comparisons = 0
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            comparisons += 1
            if values[i] + values[j] == target:
                found.add((min(values[i], values[j]), max(values[i], values[j])))
    return {"pairs": sorted(found), "comparisons": comparisons}


def unsorted_demo() -> dict:
    """
    Run the two-pointer steps on UNSORTED data to show why sorting matters.
    The algorithm 'trusts' that moving a pointer changes the sum in a known
    direction, and on unsorted data that trust is wrong.
    """
    values = [30, 10, 40, 20, 50]
    target = 60
    wrong = two_pointer_pairs(values, target, require_sorted=False)["pairs"]
    wrong = [tuple(sorted(pair)) for pair in wrong]
    correct = brute_force_pairs(values, target)["pairs"]
    missed = [pair for pair in correct if pair not in wrong]
    return {"values": values, "target": target, "two_pointer_result": wrong,
            "correct_result": correct, "missed": missed}
