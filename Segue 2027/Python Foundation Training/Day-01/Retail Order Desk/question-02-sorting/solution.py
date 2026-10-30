import random
import time


def bubble_sort_desc(values):
    result = values.copy()
    n = len(result)

    for i in range(n):
        swapped = False

        for j in range(n - 1 - i):
            if result[j] < result[j + 1]:
                result[j], result[j + 1] = result[j + 1], result[j]
                swapped = True

        if not swapped:
            break

    return result


def merge_sort_desc(values):
    if len(values) <= 1:
        return values.copy()

    mid = len(values) // 2
    left = merge_sort_desc(values[:mid])
    right = merge_sort_desc(values[mid:])

    merged = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] >= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1

    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def measure(sort_fn, values):
    start = time.perf_counter()
    result = sort_fn(values)
    elapsed = time.perf_counter() - start
    return result, elapsed


if __name__ == "__main__":
    random.seed(42)
    amounts = [random.randint(100, 100000) for _ in range(10_000)]

    bubble, bubble_time = measure(bubble_sort_desc, amounts)
    merge, merge_time = measure(merge_sort_desc, amounts)

    builtin_input = amounts.copy()
    start = time.perf_counter()
    builtin_input.sort(reverse=True)
    builtin_time = time.perf_counter() - start

    assert bubble == merge == builtin_input

    print(f"Bubble Sort: {bubble_time:.6f} seconds")
    print(f"Merge Sort:  {merge_time:.6f} seconds")
    print(f"Built-in:    {builtin_time:.6f} seconds")
    print("All algorithms returned the same sorted values.")
