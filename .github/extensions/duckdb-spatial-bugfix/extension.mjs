// Extension: duckdb-spatial-bugfix
// Documents the DuckDB OGC:CRS84 vs EPSG:4326 geometry type mismatch bug and workaround

import { joinSession } from "@github/copilot-sdk/extension";

const GITHUB_ISSUE_URL =
    "https://github.com/duckdb/duckdb/issues/22433";

const BINDER_ERROR_PATTERN = "Cannot cast GEOMETRY with CRS";
const TRANSACTION_ERROR_PATTERN =
    "TransactionContext::ActiveTransaction called without active transaction";

// Full knowledge doc returned by the tool
const KNOWLEDGE = `
# DuckDB Bug: OGC:CRS84 vs EPSG:4326 Geometry Type Mismatch in CASE Expressions

## GitHub Issue
${GITHUB_ISSUE_URL}

## DuckDB Version Observed
1.5.2  |  spatial extension dc1996b

## Affected SQL Pattern
Any CASE expression that selects geometry from two GeoParquet files where one omits
the CRS field (OGC:CRS84 default per GeoParquet 1.0 spec) and the other has an
explicit EPSG:4326 PROJJSON:

    SELECT
        CASE
            WHEN o.overture_id IS NOT NULL THEN o.geometry   -- GEOMETRY('OGC:CRS84')
            ELSE f.geometry                                   -- GEOMETRY('EPSG:4326')
        END AS geometry                                       -- Binder Error raised here
    FROM read_parquet('overture.parquet') o
    FULL OUTER JOIN read_parquet('fema.parquet') f ON ...

## Error Messages

### Primary Error (Binder Error — occurs at query-plan time)
    Binder Error: Cannot cast GEOMETRY with CRS 'OGC:CRS84' to GEOMETRY with different
    CRS 'EPSG:4326' without specifying allow_override = true.

### Cascading Error (may follow the Binder Error in the same Python process)
    INTERNAL Error: TransactionContext::ActiveTransaction called without active
    transaction. This error signals an assertion failure within DuckDB.

The cascade occurs when a prior Binder Error corrupts DuckDB's spatial extension global
state. All subsequent connections in the same process then fail, even if the query itself
is correct. Workaround: always restart the Python process after a Binder Error, or
better — avoid the bug entirely (see workaround below).

## Root Cause
GeoParquet 1.0 spec says 'crs' SHOULD be omitted when the CRS is OGC:CRS84.
Overture Maps follows this convention → DuckDB reads as GEOMETRY('OGC:CRS84').
FEMA USA Structures includes explicit EPSG:4326 PROJJSON → DuckDB reads as
GEOMETRY('EPSG:4326'). DuckDB refuses to unify them in a CASE even though both are
WGS84 with identical lon/lat WKB encoding — there is literally nothing to cast.

## Workaround (implemented in flows/produce_gold_layer.py)
Run the JOIN without geometry, read geometry via geopandas (PyArrow-based, bypasses
DuckDB spatial entirely), then merge geometry in Python:

    # Step 1: FULL OUTER JOIN — geometry columns excluded
    con = duckdb.connect()
    con.execute("LOAD spatial")
    join_df = con.sql("""
        SELECT
            COALESCE(o.overture_id, 'fema_' || f.build_id) AS building_id,
            o.overture_id, f.build_id AS fema_id
            -- include all non-geometry columns
        FROM read_parquet(overture_path) o
        FULL OUTER JOIN read_parquet(bridge_path) b ON o.overture_id = b.overture_id
        FULL OUTER JOIN read_parquet(fema_path)   f ON b.fema_id = f.build_id
    """).df()
    con.close()

    # Step 2: Read geometry via geopandas (no DuckDB spatial involved at all)
    overture_geom = (
        gpd.read_parquet(overture_path, columns=["overture_id", "geometry"])
        .to_crs("EPSG:4326").set_index("overture_id")["geometry"]
    )
    fema_geom = (
        gpd.read_parquet(fema_path, columns=["BUILD_ID", "geometry"])  # original GDB uppercase
        .rename(columns={"BUILD_ID": "build_id"})
        .to_crs("EPSG:4326").set_index("build_id")["geometry"]
    )

    # Step 3: Merge in Python — prefer Overture, fall back to FEMA
    join_df = join_df.merge(overture_geom.rename("g1"), left_on="overture_id", right_index=True, how="left")
    join_df = join_df.merge(fema_geom.rename("g2"),     left_on="fema_id",     right_index=True, how="left")
    has_ov = join_df["overture_id"].notna()
    join_df["geometry"] = pd.concat([
        join_df.loc[has_ov, "g1"],
        join_df.loc[~has_ov, "g2"],
    ]).reindex(join_df.index)
    join_df = join_df.drop(columns=["g1", "g2"])
    gdf = gpd.GeoDataFrame(join_df, geometry="geometry", crs="EPSG:4326")

## Key Notes
- FEMA parquet stores BUILD_ID in uppercase (original GDB column name). DuckDB
  auto-lowercases it in queries; geopandas/PyArrow preserves original case.
  Use columns=["BUILD_ID", "geometry"] when calling gpd.read_parquet(), then rename.
- geopandas.read_parquet() uses PyArrow and does NOT invoke DuckDB spatial at all.
  It is safe even after a Binder Error occurred in the same process.
- MRE: scripts/mre_duckdb_geometry_join.py (self-contained, creates its own test data)
`;

const session = await joinSession({
    tools: [
        {
            name: "duckdb_geometry_crs_bug",
            description:
                "Returns documented knowledge about the DuckDB OGC:CRS84 vs EPSG:4326 " +
                "geometry type mismatch bug (DuckDB 1.5.2, spatial extension dc1996b). " +
                "Use when you see: (1) 'Cannot cast GEOMETRY with CRS OGC:CRS84 to GEOMETRY " +
                "with different CRS EPSG:4326', or (2) 'TransactionContext::ActiveTransaction " +
                "called without active transaction' from code that reads GeoParquet geometry. " +
                "Also use when writing a DuckDB FULL OUTER JOIN between Overture Maps parquet " +
                "and FEMA/other EPSG:4326 GeoParquet where geometry appears in a CASE expression.",
            parameters: { type: "object", properties: {} },
            skipPermission: true,
            handler: async (_args, _inv) => KNOWLEDGE,
        },
    ],

    hooks: {
        onUserPromptSubmitted: async (input) => {
            const p = input.prompt;
            const hasBug =
                p.includes(BINDER_ERROR_PATTERN) ||
                p.includes(TRANSACTION_ERROR_PATTERN);
            const isSpatialJoin =
                /duckdb/i.test(p) &&
                /(geometry|geoparquet|overture|fema|spatial)/i.test(p);

            if (hasBug || isSpatialJoin) {
                return {
                    additionalContext:
                        "This project has a known DuckDB bug when joining GeoParquet files " +
                        "where one uses the GeoParquet 1.0 OGC:CRS84 default (no 'crs' field) " +
                        "and another has explicit EPSG:4326. Call the duckdb_geometry_crs_bug " +
                        "tool for the full documented pattern and workaround.",
                };
            }
        },
    },
});
