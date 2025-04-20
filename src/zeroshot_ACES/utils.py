from pathlib import Path
from hashlib import sha256

def get_in_out_fps(trajectories_dir: Path, output_dir: Path):
    """
    Get the input and output file paths for the trajectories.

    Args:
        trajectories_dir: Directory containing the input trajectory files.
        output_dir: Directory where the output files will be saved.

    Returns:
        A list of tuples containing the input and output file paths.
    """
    return = [output_dir / fp.relative_to(trajectories_dir) for fp in trajectories_dir.rglob("*.parquet")]

def hash_based_seed(seed: int | None, worker: int | None) -> int:
    """Generates a hash-based seed for reproducibility.

    This function generates a hash-based seed using the provided seed and worker value.

    Args:
        seed: The original seed value. THIS WILL NOT OVERWRITE THE OUTPUT. Rather, this just ensures the
            sequence of seeds chosen can be deterministically updated by changing a base parameter.
        worker: The worker identifier.

    Returns:
        A hash-based seed value.

    Examples:
        >>> hash_based_seed(42, 0)
        >>> hash_based_seed(None, 1)
        >>> hash_based_seed(1, None)
        >>> hash_based_seed(None, None)
    """

    hash_str = f"{seed}_{worker}"
    return int(sha256(hash_str.encode()).hexdigest(), 16) % (2**32 - 1)
