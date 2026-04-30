"""Thin CLI wrapper — delegates to dantinox.cli."""
from dantinox.cli import main

if __name__ == "__main__":
    import sys
    main(["train"] + sys.argv[1:])
