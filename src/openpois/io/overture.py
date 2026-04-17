#   -------------------------------------------------------------
#   Copyright (c) Henry Spatial Analysis. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   -------------------------------------------------------------

"""
This module downloads a current/latest Overture Maps Places snapshot for the
US + Puerto Rico, filtered to a set of taxonomy categories.

It is broken into the following functions:

- get_latest_release_date: Finds the most recent Overture release by listing S3.
- build_overture_s3_path: Constructs the S3 glob path for a given release.
- download_overture_snapshot: Queries S3 via DuckDB with a coarse bbox
    prefilter, decodes the result into a GeoDataFrame, applies an exact
    polygon filter against the US+PR boundary, and writes a GeoParquet file.

Spatial filter strategy (two-stage):

1. DuckDB ``WHERE`` uses predicate pushdown on Overture's ``bbox`` struct
   column, OR-ing across one or more coarse bboxes. Multiple bboxes are
   required to capture the Alaskan Near Islands, which sit at positive
   longitudes (+172 E) while the rest of the US sits at negative longitudes.
2. After the GeoDataFrame is built in memory, a spatial-join ``within``
   check against the dissolved US+PR polygon drops Canadian and Mexican
   border slivers and anything else outside the actual US+PR footprint.

Data source: s3://overturemaps-us-west-2/release/ (public, no auth required).

Category filtering uses the `taxonomy` array field. The first element
(taxonomy[1] in SQL 1-based indexing) is the L0 category. The deprecated
`categories.primary` field must NOT be used; it is removed in June 2026.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import duckdb
import geopandas as gpd
import requests


# -----------------------------------------------------------------------------
# Release discovery
# -----------------------------------------------------------------------------


def get_latest_release_date(
    bucket: str,
) -> str:
    """
    Finds the most recent Overture Maps release date by listing the S3 bucket.

    Queries the public S3 HTTP API for prefix listings under the 'release/'
    key and returns the lexicographically largest date string found.

    Args:
        bucket: The S3 bucket name hosting Overture releases.

    Returns:
        Release date string in the format 'YYYY-MM-DD.N' as it appears in S3
        (e.g., '2026-02-18.0').

    Raises:
        requests.HTTPError: If the S3 list request fails.
        ValueError: If no release prefixes are found in the bucket.
    """
    s3_list_url = (
        f"https://{bucket}.s3.amazonaws.com"
        "/?list-type=2&prefix=release%2F&delimiter=%2F"
    )
    resp = requests.get(s3_list_url, timeout = 30)
    resp.raise_for_status()

    root = ET.fromstring(resp.text)
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    prefixes = [
        el.text.rstrip("/").removeprefix("release/")
        for el in root.findall(".//s3:CommonPrefixes/s3:Prefix", ns)
    ]

    if not prefixes:
        raise ValueError(
            f"No release prefixes found in s3://{bucket}/release/. "
            "Check that the bucket is accessible."
        )

    return sorted(prefixes)[-1]


def build_overture_s3_path(
    release_date: str,
    bucket: str,
) -> str:
    """
    Returns the S3 glob path for all Places Parquet files in a given release.

    Args:
        release_date: Release identifier as returned by get_latest_release_date
            (e.g., '2026-02-18.0').
        bucket: The S3 bucket name.

    Returns:
        S3 path string suitable for DuckDB ``read_parquet()``, e.g.
        ``s3://overturemaps-us-west-2/release/2026-02-18.0/theme=places/type=place/``
    """
    return (
        f"s3://{bucket}/release/{release_date}"
        "/theme=places/type=place/*.parquet"
    )


# -----------------------------------------------------------------------------
# Download
# -----------------------------------------------------------------------------


def _build_bbox_predicate(coarse_bboxes: list[dict]) -> str:
    """Return a SQL fragment ORing Overture bbox-struct predicates together."""
    if not coarse_bboxes:
        raise ValueError("coarse_bboxes must contain at least one bbox.")
    terms = [
        (
            f"(bbox.xmin >= {b['xmin']} AND bbox.xmax <= {b['xmax']}"
            f" AND bbox.ymin >= {b['ymin']} AND bbox.ymax <= {b['ymax']})"
        )
        for b in coarse_bboxes
    ]
    return "(" + " OR ".join(terms) + ")"


def _build_taxonomy_predicate(
    allowlist: list[tuple[str, str | None]],
) -> str:
    """Return a SQL fragment ORing per-(L0, L1) taxonomy predicates together.

    If the L1 entry is ``None`` the predicate matches any L1 under that L0.
    """
    if not allowlist:
        raise ValueError("taxonomy allowlist must contain at least one entry.")
    terms = []
    for entry in allowlist:
        l0, l1 = entry[0], entry[1]
        if l1 is None:
            terms.append(f"taxonomy.hierarchy[1] = '{l0}'")
        else:
            terms.append(
                f"(taxonomy.hierarchy[1] = '{l0}' "
                f"AND taxonomy.hierarchy[2] = '{l1}')"
            )
    return "(" + " OR ".join(terms) + ")"


def download_overture_snapshot(
    output_path: Path,
    taxonomy_allowlist: list,
    boundary_gdf: gpd.GeoDataFrame,
    coarse_bboxes: list[dict],
    bucket: str,
    s3_region: str,
    release_date: str | None = None,
    source_label: str = "overture",
) -> gpd.GeoDataFrame:
    """
    Downloads filtered Overture Maps Places data and saves it as GeoParquet.

    Uses DuckDB with the httpfs and spatial extensions to query the public
    Overture Maps S3 bucket directly. Applies a two-stage spatial filter:

    1. Predicate pushdown on Overture's ``bbox`` struct using one or more
       coarse bboxes (OR-ed). Multiple bboxes allow callers to capture the
       Alaskan Near Islands (+172 E) without scanning the whole planet.
    2. A Python-side ``within`` check against ``boundary_gdf`` to keep only
       points inside the exact US+PR polygon.

    The geometry column in the source data is WKB-encoded. This function
    decodes it into a proper GeoPandas geometry column and sets the CRS to
    EPSG:4326 before applying the polygon filter and saving.

    Args:
        output_path: Path to write the output GeoParquet file.
        taxonomy_allowlist: List of (L0, L1) pairs specifying which taxonomy
            branches to retain. ``L1 = None`` means "all L1s under this L0".
            Accepts pairs as two-element tuples or lists (YAML).
            Valid L0 values (from S3 data as of 2026-02-18): 'food_and_drink',
            'shopping', 'arts_and_entertainment', 'sports_and_recreation',
            'health_care', 'services_and_business',
            'travel_and_transportation', 'lifestyle_services', 'education',
            'community_and_government', 'cultural_and_historic', 'lodging',
            'geographic_entities'.
            See: https://docs.overturemaps.org/guides/places/taxonomy/
        boundary_gdf: Single-row GeoDataFrame in EPSG:4326 containing the
            dissolved, buffered US+PR polygon. Used as the exact spatial
            filter; obtain it from ``openpois.io.boundary``.
        coarse_bboxes: List of bbox dicts (keys ``xmin, ymin, xmax, ymax``)
            used as the DuckDB predicate-pushdown prefilter. Typically
            obtained from ``openpois.io.boundary.us_pr_bboxes``.
        release_date: Overture release identifier (e.g., '2026-02-18.0').
            If None, the latest release is fetched automatically.
        bucket: S3 bucket name hosting Overture releases.
        s3_region: AWS region of the S3 bucket.
        source_label: Value for the output 'source' column.

    Returns:
        GeoDataFrame with schema:
            source (str), overture_id (str), release_date (str),
            taxonomy_l0 (str), taxonomy_l1 (str, nullable),
            taxonomy_l2 (str, nullable),
            overture_name (str, nullable), brand_name (str, nullable,
            from brand.names.primary), confidence (float64, nullable),
            geometry (Point, EPSG:4326)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if release_date is None:
        print("Detecting latest Overture release...")
        release_date = get_latest_release_date(bucket = bucket)
        print(f"Using release: {release_date}")

    s3_path = build_overture_s3_path(release_date, bucket = bucket)

    bbox_predicate = _build_bbox_predicate(coarse_bboxes)
    taxonomy_predicate = _build_taxonomy_predicate(taxonomy_allowlist)

    query = f"""
        SELECT
            '{source_label}' AS source,
            id AS overture_id,
            '{release_date}' AS release_date,
            taxonomy.hierarchy[1] AS taxonomy_l0,
            taxonomy.hierarchy[2] AS taxonomy_l1,
            taxonomy.hierarchy[3] AS taxonomy_l2,
            names.primary AS overture_name,
            brand.names.primary AS brand_name,
            confidence,
            ST_X(geometry) AS longitude,
            ST_Y(geometry) AS latitude
        FROM read_parquet('{s3_path}', hive_partitioning=1)
        WHERE
            {bbox_predicate}
            AND {taxonomy_predicate}
    """

    print(f"Querying Overture S3 at {s3_path}...")
    conn = duckdb.connect()
    conn.execute("INSTALL httpfs; LOAD httpfs;")
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute(f"SET s3_region='{s3_region}';")
    # Workaround for DuckDB httpfs bug ("Information loss on integer cast")
    # that fires on broad, wide-fanout scans of the Overture S3 bucket.
    conn.execute("SET enable_external_file_cache=false;")

    df = conn.execute(query).df()
    conn.close()

    print(f"Downloaded {len(df):,} Overture places. Building GeoDataFrame...")
    gdf = gpd.GeoDataFrame(
        df.drop(columns = ["longitude", "latitude"]),
        geometry = gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs = "EPSG:4326",
    )

    print("Applying exact US+PR polygon filter...")
    n_before = len(gdf)
    boundary_one_col = boundary_gdf[["geometry"]].to_crs(gdf.crs)
    gdf = gpd.sjoin(
        gdf, boundary_one_col, predicate = "within", how = "inner"
    ).drop(columns = "index_right").reset_index(drop = True)
    print(
        f"Polygon filter kept {len(gdf):,} of {n_before:,} "
        f"({n_before - len(gdf):,} dropped outside US+PR)."
    )

    gdf.to_parquet(output_path)
    print(f"Saved Overture snapshot to {output_path}")
    return gdf
