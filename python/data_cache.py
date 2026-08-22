"""
ARC Lens — shared CIFAR-10 cache location
==========================================

One cache dir for CIFAR-10 across all demo scripts, instead of each one
re-downloading its own copy under the repo. ``$ARC_DEMO_DATA`` still wins
when set (existing override behavior); otherwise this picks an OS-standard
cache directory.
"""

import os
import sys


def cifar_root() -> str:
    """Return the shared CIFAR-10 cache directory, creating it if needed."""
    override = os.environ.get("ARC_DEMO_DATA")
    if override:
        root = override
    elif os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        root = os.path.join(base, "arc-lens", "cifar10")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        root = os.path.join(base, "arc-lens", "cifar10")
    os.makedirs(root, exist_ok=True)
    return root


# Real CIFAR-10 tar.gz is ~170MB; a corrupted/truncated download that's wildly
# smaller is the failure mode that caused a real 30-minute stall in this repo.
_EXPECTED_TARBALL_MIN_BYTES = 150 * 1024 * 1024


def warn_if_cache_incomplete(root: str) -> None:
    """Print a loud stderr warning if the cache looks present-but-broken.

    ponytail: heuristic size/existence check only, not a checksum — good
    enough to catch the truncated-download incident; upgrade to
    torchvision's _check_integrity() if silent corruption recurs.
    """
    tarball = os.path.join(root, "cifar-10-python.tar.gz")
    batches_dir = os.path.join(root, "cifar-10-batches-py")
    has_tarball = os.path.isfile(tarball)
    has_batches = os.path.isdir(batches_dir)

    if not has_tarball and not has_batches:
        return  # true cache miss — download() will print its own progress

    tarball_ok = has_tarball and os.path.getsize(tarball) >= _EXPECTED_TARBALL_MIN_BYTES
    if has_batches or tarball_ok:
        return  # looks complete, nothing to warn about

    print(
        f"[arc-lens] CIFAR-10 cache miss or incomplete at {root} — "
        "downloading ~170MB, this can take minutes",
        file=sys.stderr,
    )


def demo() -> None:
    root = cifar_root()
    assert os.path.isdir(root), "cifar_root() must create the directory"
    assert root  # non-empty path
    warn_if_cache_incomplete(root)  # should be a no-op / silent on empty dir
    print(f"cifar_root() -> {root}")


if __name__ == "__main__":
    demo()
