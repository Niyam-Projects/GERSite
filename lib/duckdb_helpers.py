"""
DuckDB connection factory and storage path resolution for GERSite.

Provides:
- ``get_connection`` — DuckDB connection with spatial + httpfs extensions loaded.
- ``StorageConfig`` — resolves local or S3 paths from config.gers.yaml.
- ``aoi_bbox_filter`` — DuckDB WHERE clause fragment for a named AOI bbox.

Pattern adapted from src/openpois/io/overture.py.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import duckdb
import yaml


# ---------------------------------------------------------------------------
# DuckDB connection factory
# ---------------------------------------------------------------------------


def get_connection(
    memory_limit: str = "8GB",
    threads: int = 4,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection with spatial and httpfs extensions loaded.

    Args:
        memory_limit: DuckDB memory cap (e.g. '4GB', '16GB').
        threads: Number of DuckDB worker threads.
        read_only: If True, open the connection in read-only mode.

    Returns:
        Configured DuckDB connection.
    """
    con = duckdb.connect(database=":memory:", read_only=read_only)
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("INSTALL spatial; LOAD spatial")
    con.execute("INSTALL httpfs; LOAD httpfs")
    return con


# ---------------------------------------------------------------------------
# Storage configuration
# ---------------------------------------------------------------------------


@dataclass
class StorageConfig:
    """Resolves GERSite storage paths from config.gers.yaml.

    Paths are resolved relative to ``root``.  Swap ``root`` to an S3 prefix
    (e.g. ``s3://your-bucket/gers/``) to enable cloud-native operation.

    Attributes:
        root: Base storage directory or S3 prefix.
        bronze: Bronze-layer sub-paths (raw source data).
        silver: Silver-layer sub-paths (bridge files).
        gold: Gold-layer sub-paths (unified output).
    """

    root: str
    bronze: dict = field(default_factory=dict)
    silver: dict = field(default_factory=dict)
    gold: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, config_path: str | Path = "config.gers.yaml") -> "StorageConfig":
        """Load from a config.gers.yaml file."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        storage = cfg["storage"]
        root = os.path.expanduser(storage["root"])
        return cls(
            root=root,
            bronze=storage.get("bronze", {}),
            silver=storage.get("silver", {}),
            gold=storage.get("gold", {}),
        )

    def resolve(self, *parts: str) -> str:
        """Join root with sub-path parts and expand ~ if present.

        Works for both local paths (returns absolute string) and S3 paths
        (returns s3://bucket/key string unchanged).
        """
        joined = "/".join([self.root.rstrip("/"), *parts])
        if joined.startswith("s3://"):
            return joined
        return str(Path(os.path.expanduser(joined)))

    def bronze_path(self, key: str) -> str:
        """Resolve a named bronze sub-path, e.g. 'overture_buildings'."""
        return self.resolve(self.bronze[key])

    def silver_path(self, key: str) -> str:
        """Resolve a named silver sub-path, e.g. 'fema_bridge'."""
        return self.resolve(self.silver[key])

    def gold_path(self, key: str, aoi: Optional[str] = None) -> str:
        """Resolve a named gold sub-path, optionally scoped to an AOI.

        Args:
            key: Config key under storage.gold (e.g. 'buildings').
            aoi: Optional AOI name to append as a sub-directory.
        """
        base = self.resolve(self.gold[key])
        if aoi:
            return f"{base}/{aoi}"
        return base


# ---------------------------------------------------------------------------
# AOI helpers
# ---------------------------------------------------------------------------


def load_aoi_config(config_path: str | Path = "config.gers.yaml") -> dict:
    """Return the AOI dict from config.gers.yaml.

    Returns:
        Dict mapping AOI name → {label, bbox, geojson}.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg["aoi"]


def aoi_bbox(aoi_name: str, config_path: str | Path = "config.gers.yaml") -> list[float]:
    """Return [xmin, ymin, xmax, ymax] for the named AOI.

    Args:
        aoi_name: One of 'saipan', 'guam', 'puerto_rico', 'miami_dade'.
        config_path: Path to config.gers.yaml.

    Returns:
        Bounding box as [xmin, ymin, xmax, ymax] in WGS84.

    Raises:
        KeyError: If aoi_name is not found in config.
    """
    aois = load_aoi_config(config_path)
    if aoi_name not in aois:
        raise KeyError(
            f"AOI '{aoi_name}' not found. Available: {list(aois.keys())}"
        )
    return aois[aoi_name]["bbox"]


def aoi_bbox_sql(aoi_name: str, config_path: str | Path = "config.gers.yaml") -> str:
    """Return a DuckDB WHERE fragment for the named AOI bounding box.

    Assumes a geometry column named ``geometry`` in EPSG:4326.  Use
    ``ST_Intersects(geometry, ST_MakeEnvelope(...))`` for polygon columns or
    ``ST_X(geometry) BETWEEN ...`` for point columns.

    Args:
        aoi_name: AOI name (e.g. 'saipan').
        config_path: Path to config.gers.yaml.

    Returns:
        SQL WHERE clause fragment, e.g.:
        "ST_Intersects(geometry, ST_MakeEnvelope(145.65, 15.06, 145.88, 15.32))"
    """
    xmin, ymin, xmax, ymax = aoi_bbox(aoi_name, config_path)
    return (
        f"ST_Intersects(geometry, "
        f"ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}))"
    )


def aoi_bbox_struct_filter(
    aoi_name: str,
    config_path: str | Path = "config.gers.yaml",
) -> str:
    """Return a DuckDB predicate for Overture's ``bbox`` struct column.

    Overture GeoParquet uses a ``bbox`` struct with keys
    (xmin, ymin, xmax, ymax) to enable predicate pushdown before
    loading full geometries.

    Returns:
        SQL fragment like:
        "bbox.xmin <= 145.88 AND bbox.xmax >= 145.65 AND ..."
    """
    xmin, ymin, xmax, ymax = aoi_bbox(aoi_name, config_path)
    return (
        f"bbox.xmin <= {xmax} AND bbox.xmax >= {xmin} "
        f"AND bbox.ymin <= {ymax} AND bbox.ymax >= {ymin}"
    )
