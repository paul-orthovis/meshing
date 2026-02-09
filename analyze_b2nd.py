#!/usr/bin/env python3
"""Load a b2nd Blosc file and print statistical information."""

import argparse
import sys

import blosc2
import matplotlib.pyplot as plt
import numpy as np


def _to_2d_slice(volume, idx):
    """Return a 2D slice from an array, collapsing leading axes if needed."""
    if volume.ndim == 2:
        return volume
    if volume.ndim >= 3:
        slice_ = volume[idx]
        while slice_.ndim > 2:
            mid = slice_.shape[0] // 2
            slice_ = slice_[mid]
        return slice_
    raise ValueError("Expected at least 2D data")


def show_slices(data, n_slices=12):
    if data.ndim < 2:
        return

    if data.ndim == 4:
        assert data.shape[0] == 1
        data = data[0]

    vmin = np.min(data)
    vmax = np.max(data)

    if data.ndim == 2:
        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title("Slice")
        fig.colorbar(im, ax=ax)
    else:
        n_slices = min(n_slices, data.shape[0])
        indices = np.linspace(0, data.shape[0] - 1, n_slices, dtype=int)
        n_cols = int(np.ceil(np.sqrt(n_slices)))
        n_rows = int(np.ceil(n_slices / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        axes = np.atleast_1d(axes).ravel()

        for ax, idx in zip(axes, indices):
            slice_2d = _to_2d_slice(data, idx)
            im = ax.imshow(slice_2d, cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(f"z={idx}")
            fig.colorbar(im, ax=ax)

        # Hide any unused axes
        for ax in axes[len(indices):]:
            ax.axis("off")

    plt.tight_layout()
    plt.show()


def analyze_b2nd(filepath):
    """Load a b2nd file, print statistics, and show example slices."""
    try:
        array = blosc2.open(filepath)
    except Exception as e:
        print(f"Error loading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    data = array[:]
    
    min_val = np.min(data)
    max_val = np.max(data)
    mean_val = np.mean(data)
    std_val = np.std(data)
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    
    print(f"File: {filepath}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print()
    print(f"Min: {min_val}")
    print(f"Max: {max_val}")
    print(f"Mean: {mean_val:.6f}")
    print(f"Std Dev: {std_val:.6f}")
    print(f"Q1 (25th percentile): {q25:.6f}")
    print(f"Q2 (50th percentile/Median): {q50:.6f}")
    print(f"Q3 (75th percentile): {q75:.6f}")

    show_slices(data)


def main():
    parser = argparse.ArgumentParser(
        description="Load a b2nd Blosc file and print statistics"
    )
    parser.add_argument(
        "filepath",
        help="Path to the b2nd file"
    )
    args = parser.parse_args()
    
    analyze_b2nd(args.filepath)


if __name__ == "__main__":
    main()
