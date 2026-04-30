"""Thin CLI wrapper — delegates to dantinox.cli."""
from dantinox.cli import main

if __name__ == "__main__":
    import sys
    main(["benchmark"] + sys.argv[1:])
