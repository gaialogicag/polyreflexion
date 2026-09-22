"""Command-line entry points.

``polyrx-bench`` is Hydra-driven and is the entry point for anything that
produces a published number. The others are small argparse tools that compose
the same config tree through :func:`polyreflexion.cli.compose.load_config`.
"""
