"""Dataset adapters: one interface, any benchmark."""

from polyrx.datasets.base import (
    DatasetAdapter,
    Item,
    available_adapters,
    get_adapter,
    register_adapter,
    sample_by_group,
)

__all__ = [
    "DatasetAdapter",
    "Item",
    "available_adapters",
    "get_adapter",
    "register_adapter",
    "sample_by_group",
]
