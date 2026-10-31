"""
bst_catalogue.py
================
Module 1 - Product Category Binary Search Tree (BST), written from scratch.

Business problem
    The platform has many product categories and wants faster category
    browsing and searching than scanning an unordered list.

How a BST helps
    Every node keeps smaller names in its LEFT subtree and larger names in its
    RIGHT subtree, so each comparison lets us ignore about half of what is left.

Nothing in this file uses a built-in sorted collection (no sorted(), no bisect,
no sortedcontainers). Only Node objects and string comparisons are used.
"""

from typing import Optional

# 15 sample categories, listed in the order they are inserted.
# This order was chosen so the sample tree is nicely balanced (height 4).
SAMPLE_CATEGORIES = [
    "Grocery", "Clothing", "Music", "Beauty", "Furniture",
    "Home & Kitchen", "Sports", "Automotive", "Books", "Electronics",
    "Garden", "Health", "Jewelry", "Pet Supplies", "Toys",
]


def clean_name(text: str) -> str:
    """Trim the text and collapse repeated spaces: '  home   decor ' -> 'home decor'."""
    return " ".join(text.split())


def _key(text: str) -> str:
    """Comparison key. Case-insensitive, so 'books' and 'Books' are the same category."""
    return text.casefold()


class Node:
    """One category in the tree."""

    def __init__(self, value: str):
        self.value = value                  # the category name
        self.left: Optional["Node"] = None   # subtree of names that come BEFORE this one
        self.right: Optional["Node"] = None  # subtree of names that come AFTER this one


def _step(number: int, node_value: str, comparison: str, decision: str) -> dict:
    """One row of the step-by-step dry run table."""
    return {
        "Step": number,
        "Node visited": node_value,
        "Comparison": comparison,
        "Decision": decision,
    }


class CategoryBST:
    """A Binary Search Tree of category names."""

    def __init__(self):
        self.root: Optional[Node] = None
        self.size = 0  # number of nodes

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------
    def insert(self, value: str):
        """
        Insert a category.

        Returns (inserted, path, steps)
            inserted : True if added, False if it already existed
            path     : category names visited (the new node is last when inserted)
            steps    : rows for the dry-run table
        """
        value = clean_name(value)
        if not value:
            raise ValueError("Category name cannot be empty.")

        # Case 1: empty tree -> the new node becomes the root.
        if self.root is None:
            self.root = Node(value)
            self.size += 1
            steps = [_step(1, "(empty tree)", "-", f"Tree is empty, so '{value}' becomes the root")]
            return True, [value], steps

        # Case 2: walk down from the root until we find the empty spot.
        path, steps = [], []
        current = self.root
        while True:
            path.append(current.value)
            number = len(steps) + 1

            if _key(value) == _key(current.value):
                steps.append(_step(number, current.value, f"'{value}' = '{current.value}'",
                                   "Already in the tree, nothing inserted"))
                return False, path, steps

            go_left = _key(value) < _key(current.value)
            sign, side = ("<", "left") if go_left else (">", "right")
            comparison = f"'{value}' {sign} '{current.value}'"
            child = current.left if go_left else current.right

            if child is None:  # empty spot found -> attach the new node here
                new_node = Node(value)
                if go_left:
                    current.left = new_node
                else:
                    current.right = new_node
                self.size += 1
                path.append(value)
                steps.append(_step(number, current.value, comparison,
                                   f"Go {side}: empty spot, insert '{value}' here"))
                return True, path, steps

            steps.append(_step(number, current.value, comparison, f"Go {side}"))
            current = child

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, value: str):
        """
        Search for a category.

        Returns (found, path, steps) - same shape as insert().
        """
        value = clean_name(value)
        if not value:
            raise ValueError("Category name cannot be empty.")

        if self.root is None:
            return False, [], [_step(1, "(empty tree)", "-", "Tree is empty, category not found")]

        path, steps = [], []
        current = self.root
        while current is not None:
            path.append(current.value)
            number = len(steps) + 1

            if _key(value) == _key(current.value):
                steps.append(_step(number, current.value, f"'{value}' = '{current.value}'", "Found it!"))
                return True, path, steps

            go_left = _key(value) < _key(current.value)
            sign, side = ("<", "left") if go_left else (">", "right")
            comparison = f"'{value}' {sign} '{current.value}'"
            child = current.left if go_left else current.right

            if child is None:
                steps.append(_step(number, current.value, comparison,
                                   f"Go {side}: nothing there, so '{value}' is not in the tree"))
                return False, path, steps

            steps.append(_step(number, current.value, comparison, f"Go {side}"))
            current = child

        return False, path, steps  # not reached, kept for safety

    # ------------------------------------------------------------------
    # Traversals (each visits every node exactly once)
    # ------------------------------------------------------------------
    def inorder(self) -> list:
        """Left -> Node -> Right. For a BST this gives alphabetical order."""
        result = []

        def visit(node):
            if node is None:
                return
            visit(node.left)
            result.append(node.value)
            visit(node.right)

        visit(self.root)
        return result

    def preorder(self) -> list:
        """Node -> Left -> Right. Parents come before children."""
        result = []

        def visit(node):
            if node is None:
                return
            result.append(node.value)
            visit(node.left)
            visit(node.right)

        visit(self.root)
        return result

    def postorder(self) -> list:
        """Left -> Right -> Node. Children come before parents."""
        result = []

        def visit(node):
            if node is None:
                return
            visit(node.left)
            visit(node.right)
            result.append(node.value)

        visit(self.root)
        return result

    # ------------------------------------------------------------------
    # Helpers for display
    # ------------------------------------------------------------------
    def height(self) -> int:
        """Number of levels in the tree (0 for an empty tree)."""

        def measure(node):
            if node is None:
                return 0
            return 1 + max(measure(node.left), measure(node.right))

        return measure(self.root)

    def layout(self) -> list:
        """
        Coordinates for drawing the tree.
        x = position in alphabetical (inorder) order, depth = level (root is 0).
        """
        items = []
        counter = [0]

        def walk(node, depth, parent):
            if node is None:
                return
            walk(node.left, depth + 1, node.value)
            items.append({"value": node.value, "x": counter[0], "depth": depth, "parent": parent})
            counter[0] += 1
            walk(node.right, depth + 1, node.value)

        walk(self.root, 0, None)
        return items


def build_sample_tree() -> CategoryBST:
    """A tree pre-loaded with the 15 sample categories."""
    tree = CategoryBST()
    for category in SAMPLE_CATEGORIES:
        tree.insert(category)
    return tree
