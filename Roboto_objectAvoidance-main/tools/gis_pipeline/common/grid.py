"""Re-export of the shared frame/grid primitives.

The implementation lives in `roboto_core.frames` because the ROS nodes need
it too, and the coordinate-frame contract only holds if there is exactly one
implementation of it. Importing it from here keeps the GIS pipeline's import
paths stable.

See roboto_core/frames.py for the row-order convention -- read it before
touching anything that indexes a grid.
"""

from roboto_core.frames import (  # noqa: F401
    PGM_FREE,
    PGM_OCCUPIED,
    PGM_UNKNOWN,
    GridSpec,
    from_raster,
    load_occupancy,
    save_occupancy,
    to_raster,
)

__all__ = [
    "GridSpec",
    "from_raster",
    "to_raster",
    "save_occupancy",
    "load_occupancy",
    "PGM_OCCUPIED",
    "PGM_FREE",
    "PGM_UNKNOWN",
]
