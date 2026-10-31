"""
traffic_analysis.py
===================
Module 4 - Busiest continuous time window using a fixed-size sliding window.

Business problem
    To size servers for peak load we need the busiest continuous 3-hour period
    (or any window size), not just the single busiest hour.

How a sliding window helps
    Add up the first window once. To move one hour forward, subtract the hour
    that leaves and add the hour that enters. That is 2 operations per slide
    instead of re-adding the whole window every time.
"""

import random

# 24 sample hourly visitor counts (hour 0 = midnight ... hour 23 = 11 pm)
DEFAULT_TRAFFIC = [
    120, 90, 70, 60, 55, 80,         # 00:00 - 05:59  quiet night
    150, 260, 410, 520, 600, 640,    # 06:00 - 11:59  morning ramp-up
    700, 680, 620, 590, 640, 780,    # 12:00 - 17:59  lunch bump, afternoon
    920, 1010, 880, 640, 380, 210,   # 18:00 - 23:59  evening peak
]


def generate_random_traffic(seed=None) -> list:
    """A new day of traffic: the sample curve with +/- 20% random variation."""
    rng = random.Random(seed)
    return [int(value * rng.uniform(0.8, 1.2)) for value in DEFAULT_TRAFFIC]


def format_hour_range(start: int, window_size: int) -> str:
    """Start hour 18, size 3 -> '18:00 to 21:00' (covers hours 18, 19 and 20)."""
    return f"{start:02d}:00 to {start + window_size:02d}:00"


def _check_inputs(values: list, window_size: int) -> None:
    if len(values) == 0:
        raise ValueError("Traffic data is empty.")
    if window_size < 1 or window_size > len(values):
        raise ValueError(f"Window size must be between 1 and {len(values)}.")


def sliding_window_max(values: list, window_size: int = 3) -> dict:
    """
    Fixed-size sliding window.

    1. Calculate the first window sum.
    2. Slide one step: new sum = old sum - outgoing value + incoming value.
    3. Track the maximum sum and where its window starts.
    """
    _check_inputs(values, window_size)
    n = len(values)

    # Step 1: first window, added up one by one
    window_sum = 0
    for i in range(window_size):
        window_sum += values[i]

    best_sum, best_start = window_sum, 0
    window_sums = [window_sum]
    operations = window_size  # additions so far
    steps = [{
        "Start hour": 0,
        "Window (hours)": f"0 to {window_size - 1}",
        "Removed": "-",
        "Added": "-",
        "Window total": window_sum,
        "New best?": "Yes (first window)",
    }]

    # Steps 2 and 3: slide across the rest of the day
    for start in range(1, n - window_size + 1):
        outgoing = values[start - 1]                    # hour that just left the window
        incoming = values[start + window_size - 1]      # hour that just entered
        window_sum = window_sum - outgoing + incoming
        operations += 2

        window_sums.append(window_sum)
        is_new_best = window_sum > best_sum             # ">" keeps the earliest window on ties
        if is_new_best:
            best_sum, best_start = window_sum, start

        steps.append({
            "Start hour": start,
            "Window (hours)": f"{start} to {start + window_size - 1}",
            "Removed": f"-{outgoing}",
            "Added": f"+{incoming}",
            "Window total": window_sum,
            "New best?": "Yes" if is_new_best else "No",
        })

    return {
        "window_sums": window_sums,
        "best_start": best_start,
        "best_end": best_start + window_size - 1,
        "best_sum": best_sum,
        "operations": operations,
        "steps": steps,
    }


def naive_window_max(values: list, window_size: int = 3) -> dict:
    """
    The simple approach for comparison: re-add every window from scratch.
    That is (n - k + 1) windows x k additions each = O(n * k).
    """
    _check_inputs(values, window_size)
    n = len(values)
    best_sum, best_start, additions = None, 0, 0

    for start in range(n - window_size + 1):
        total = 0
        for offset in range(window_size):
            total += values[start + offset]
            additions += 1
        if best_sum is None or total > best_sum:
            best_sum, best_start = total, start

    return {"best_start": best_start, "best_end": best_start + window_size - 1,
            "best_sum": best_sum, "operations": additions}
