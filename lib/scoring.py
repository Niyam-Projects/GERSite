"""
Confidence scoring and risk flagging for GERSite Gold layer buildings.

Each building in the Gold layer receives a ``conflation_confidence`` score
based on source agreement, and optionally an ``nsi_risk_flag`` when an
NSI occupancy record exists but no building footprint is in Overture or FEMA.

Scoring model:
    1.0 — High:    Building in Overture AND FEMA with IoU >= high_iou_threshold
    0.6 — Medium:  Building in Overture only (no FEMA match)
    0.3 — Low:     FEMA-only candidate feature (no Overture match)
    NSI risk flag: NSI match exists but building has no Overture or FEMA footprint.

OSM columns are reserved (null) in this phase; confidence will be updated
when the Overture-OSM bridge is activated in a future deployment phase.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

CONFIDENCE_HIGH = 1.0
CONFIDENCE_MEDIUM = 0.6
CONFIDENCE_LOW = 0.3


def compute_confidence(
    has_overture: np.ndarray,
    has_fema: np.ndarray,
    max_iou: np.ndarray,
    high_iou_threshold: float = 0.80,
) -> np.ndarray:
    """Compute per-building conflation confidence scores.

    Vectorized; operates on boolean/float arrays aligned by row index.

    Scoring rules (evaluated in priority order):
      1. has_overture AND has_fema AND max_iou >= high_iou_threshold → 1.0 (High)
      2. has_overture AND NOT has_fema → 0.6 (Medium)
      3. NOT has_overture AND has_fema → 0.3 (Low / candidate)
      4. has_overture AND has_fema AND max_iou < high_iou_threshold → 0.6
         (Overture + FEMA but weak overlap: treat as Overture-anchored Medium)

    Args:
        has_overture: Boolean array — True if building has an Overture record.
        has_fema: Boolean array — True if building has a FEMA bridge match.
        max_iou: Float array — max IoU score from fema_bridge (0 if no match).
        high_iou_threshold: IoU cutoff for High confidence (default 0.80).

    Returns:
        Float array of confidence scores aligned with input arrays.

    Example:
        >>> has_ov = np.array([True, True, False])
        >>> has_fe = np.array([True, False, True])
        >>> iou    = np.array([0.90, 0.0,  0.05])
        >>> compute_confidence(has_ov, has_fe, iou)
        array([1.0, 0.6, 0.3])
    """
    has_overture = np.asarray(has_overture, dtype=bool)
    has_fema = np.asarray(has_fema, dtype=bool)
    max_iou = np.asarray(max_iou, dtype=float)

    scores = np.full(len(has_overture), CONFIDENCE_MEDIUM, dtype=float)

    # Low confidence: FEMA-only candidates
    fema_only = ~has_overture & has_fema
    scores[fema_only] = CONFIDENCE_LOW

    # High confidence: Overture + FEMA with strong IoU agreement
    high = has_overture & has_fema & (max_iou >= high_iou_threshold)
    scores[high] = CONFIDENCE_HIGH

    return scores


# ---------------------------------------------------------------------------
# NSI risk flagging
# ---------------------------------------------------------------------------


def flag_nsi_risk(
    has_nsi_match: np.ndarray,
    has_overture: np.ndarray,
    has_fema: np.ndarray,
) -> np.ndarray:
    """Flag buildings where NSI has a record but no footprint source exists.

    An NSI record indicates an occupancy/value estimate for a structure.
    If NSI has a match but neither Overture nor FEMA provides a footprint,
    the building is flagged for manual review.

    Args:
        has_nsi_match: Boolean array — True if building has an NSI bridge match.
        has_overture: Boolean array — True if building has an Overture record.
        has_fema: Boolean array — True if building has a FEMA bridge match.

    Returns:
        Boolean array — True where NSI risk flag should be set.

    Example:
        >>> flag_nsi_risk(
        ...     has_nsi_match=np.array([True, True, False]),
        ...     has_overture= np.array([False, True,  True]),
        ...     has_fema=     np.array([False, False, True]),
        ... )
        array([ True, False, False])
    """
    has_nsi_match = np.asarray(has_nsi_match, dtype=bool)
    has_overture = np.asarray(has_overture, dtype=bool)
    has_fema = np.asarray(has_fema, dtype=bool)

    has_any_footprint = has_overture | has_fema
    return has_nsi_match & ~has_any_footprint


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def confidence_summary(scores: np.ndarray) -> dict:
    """Return a count/percentage breakdown of confidence levels.

    Args:
        scores: Array of confidence scores from ``compute_confidence``.

    Returns:
        Dict with keys 'high', 'medium', 'low', 'total' and their counts.
    """
    scores = np.asarray(scores, dtype=float)
    total = len(scores)
    high = int(np.sum(scores == CONFIDENCE_HIGH))
    medium = int(np.sum(scores == CONFIDENCE_MEDIUM))
    low = int(np.sum(scores == CONFIDENCE_LOW))
    return {
        "total": total,
        "high": high,
        "medium": medium,
        "low": low,
        "pct_high": round(high / total * 100, 1) if total else 0.0,
        "pct_medium": round(medium / total * 100, 1) if total else 0.0,
        "pct_low": round(low / total * 100, 1) if total else 0.0,
    }
