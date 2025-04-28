from hashlib import sha256
from pathlib import Path


def get_in_out_fps(trajectories_dir: Path, output_dir: Path):
    """Get the input and output file paths for the trajectories.

    Args:
        trajectories_dir: Directory containing the input trajectory files.
        output_dir: Directory where the output files will be saved.

    Returns:
        A list of tuples containing the input and output file paths. Output filepaths are simply the input
            file paths relative to the trajectories directory nested within the output directory.

    Examples:
        >>> with tempfile.TemporaryDirectory() as tempdir:
        ...     trajectories_dir = Path(tempdir) / "trajectories"
        ...     output_dir = Path(tempdir) / "output"
        ...     fp_1 = trajectories_dir / "file1.parquet"
        ...     fp_2 = trajectories_dir / "nested" / "file2.parquet"
        ...     fp_2.parent.mkdir(parents=True, exist_ok=True)
        ...     fp_1.touch()
        ...     fp_2.touch()
        ...     in_out_fps = get_in_out_fps(trajectories_dir, output_dir)
        ...     print([fp.relative_to(output_dir).as_posix() for _, fp in in_out_fps])
        ['file1.parquet', 'nested/file2.parquet']
    """
    return [(fp, output_dir / fp.relative_to(trajectories_dir)) for fp in trajectories_dir.rglob("*.parquet")]


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
        2658406739
        >>> hash_based_seed(None, 1)
        1318793722
        >>> hash_based_seed(1, None)
        2643917841
        >>> hash_based_seed(None, None)
        3685662983
    """

    hash_str = f"{seed}_{worker}"
    return int(sha256(hash_str.encode()).hexdigest(), 16) % (2**32 - 1)
