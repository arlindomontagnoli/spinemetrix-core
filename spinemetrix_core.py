"""
SpineMetrix — sagittal parameter core.

The exact functions that run in production at spinemetrix.com: this is the code that produces the
published numbers. No server, no interface, no database — vertebra marks in (corner coordinates in
image pixels), parameters out.

Coordinate convention: origin at the top-left of the image, Y growing DOWNWARDS, as in any digital
image (and as DICOM delivers it).

Usage:
    from spinemetrix_core import load_payload, compute_pelvic_parameters
    payload = load_payload(json.load(open("marks.json")))
    print(compute_pelvic_parameters(payload))

License: Apache-2.0 (see LICENSE).
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import UnivariateSpline  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - without scipy the spline falls back to linear interpolation
    UnivariateSpline = None  # type: ignore
    _HAS_SCIPY = False


def log(msg: str) -> None:
    """On the server this feeds /debug/logs; here it is silent on purpose."""
    return None


# --------------------------------------------------------------------------------------------
# Input. The server uses pydantic models; here plain objects with .x/.y and .points are enough, which
# is what lets the code below stay IDENTICAL to production — and that identity is the whole point.
# --------------------------------------------------------------------------------------------
class Point:
    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float) -> None:
        self.x = float(x)
        self.y = float(y)

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"


class FemoralHead:
    __slots__ = ("center", "radius")

    def __init__(self, center: Point, radius: float = 0.0) -> None:
        self.center = center
        self.radius = float(radius)


class Payload:
    """One marked study: groups of 4 corners per vertebra + femoral heads."""

    def __init__(self, points, femoralHeads=None, sacral_extra: int = 0,
                 image_height=None, image_width=None, analysis_mode: str = "sagital") -> None:
        self.points = points
        self.femoralHeads = femoralHeads or []
        self.sacral_extra = int(sacral_extra or 0)
        self.image_height = image_height
        self.image_width = image_width
        self.analysis_mode = analysis_mode


def load_payload(doc: Dict[str, Any]) -> Payload:
    """Accepts the .json saved by SpineMetrix (server record or bare payload)."""
    p = doc.get("payload", doc)
    groups = [[Point(pt["x"], pt["y"]) for pt in g] for g in (p.get("points") or [])]
    heads = []
    for fh in (p.get("femoralHeads") or []):
        if fh and fh.get("center"):
            heads.append(FemoralHead(Point(fh["center"]["x"], fh["center"]["y"]), fh.get("radius", 0.0)))
    return Payload(groups, heads, p.get("sacral_extra", 0),
                   p.get("image_height"), p.get("image_width"),
                   p.get("analysis_mode", "sagital"))


# --------------------------------------------------------------------------------------------
# Geometry, reference spline and endplate selection (verbatim from the production app.py)
# --------------------------------------------------------------------------------------------
def _flatten_points(payload: Payload) -> np.ndarray:
    all_pts: List[Point] = []
    for g in payload.points or []:
        all_pts.extend(g or [])
    all_pts.extend(payload.current or [])
    if not all_pts:
        return np.zeros((0, 2), dtype=float)
    arr = np.array([[float(p.x), float(p.y)] for p in all_pts], dtype=float)
    return arr


def _compute_vertebra_centroids(payload: Payload) -> List[Point]:
    """Compute per-group centroids (vertebrae) from payload.points."""
    verts: List[Point] = []
    for group in payload.points or []:
        if not group:
            continue
        garr = np.array([[float(p.x), float(p.y)] for p in group], dtype=float)
        if garr.shape[0] == 0:
            continue
        gc = garr.mean(axis=0)
        verts.append(Point(x=float(gc[0]), y=float(gc[1])))
    return verts


def _compute_s1_inferior_endplate_midpoint(payload: Payload) -> Optional[Point]:
    """Return midpoint of S1 inferior endplate (most caudal group, bottom 2 points by Y)."""
    best_arr: Optional[np.ndarray] = None
    best_cy = -np.inf

    for group in payload.points or []:
        if not group or len(group) < 2:
            continue
        garr = np.array([[float(p.x), float(p.y)] for p in group], dtype=float)
        if garr.shape[0] < 2:
            continue
        cy = float(garr[:, 1].mean())
        if cy > best_cy:
            best_cy = cy
            best_arr = garr

    if best_arr is None or best_arr.shape[0] < 2:
        return None

    # Inferior endplate in image coordinates = two largest Y values.
    sorted_by_y_desc = best_arr[np.argsort(best_arr[:, 1])[::-1]]
    p1 = sorted_by_y_desc[0]
    p2 = sorted_by_y_desc[1]
    mid = (p1 + p2) / 2.0
    return Point(x=float(mid[0]), y=float(mid[1]))


def _compute_spline_vertebra_points(payload: Payload) -> List[Point]:
    """Return vertebra centroids used to anchor the global spline reference."""
    return _compute_vertebra_centroids(payload)


def _prepare_xy_for_spline(centroids: List[Point]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (cys, cxs) sorted by Y ascending, deduplicated on Y to satisfy spline requirements."""
    if not centroids:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)
    arr = np.array([[c.x, c.y] for c in centroids], dtype=float)
    # Sort by Y (ascending: top -> bottom in image coordinates)
    idx = np.argsort(arr[:, 1])
    arr = arr[idx]
    cxs = arr[:, 0]
    cys = arr[:, 1]
    # Deduplicate exact Y values by averaging corresponding Xs
    # This avoids issues where spline expects strictly increasing X (here: Y)
    uniq_y, inv = np.unique(cys, return_inverse=True)
    if uniq_y.shape[0] != cys.shape[0]:
        # Aggregate X by unique Y
        agg_x = np.zeros_like(uniq_y)
        counts = np.zeros_like(uniq_y)
        for i, yi in enumerate(inv):
            agg_x[yi] += cxs[i]
            counts[yi] += 1
        cys = uniq_y
        cxs = agg_x / np.maximum(counts, 1)
    return cys, cxs


def _get_spline_smoothing_factor(payload: Optional[Payload] = None, image_height: Optional[float] = None) -> float:
    """Smoothing factor for UnivariateSpline: H / 80 (legacy default 20 when H is unknown)."""
    h = image_height
    if h is None and payload is not None and payload.image_height is not None:
        h = float(payload.image_height)
    if h is None or h <= 0:
        return 20.0
    return float(h) / 80.0


def _build_reference_spline_xy(
    centroids: List[Point],
    n_samples: int = 1000,
    smoothing_factor: float = 20.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Return smoothed spline samples (x_vals, y_grid, method) from centroid anchors."""
    if len(centroids) < 2:
        return None, None, "raw"

    cys, cxs = _prepare_xy_for_spline(centroids)
    if cys.shape[0] < 2:
        return None, None, "raw"

    y_grid = np.linspace(float(cys.min()), float(cys.max()), n_samples)
    try:
        if _HAS_SCIPY and UnivariateSpline is not None:
            k = 3 if cys.shape[0] >= 4 else max(1, int(cys.shape[0] - 1))
            spl = UnivariateSpline(cys, cxs, k=k)
            spl.set_smoothing_factor(smoothing_factor)
            x_vals = spl(y_grid)
            return x_vals, y_grid, "scipy.UnivariateSpline"
    except Exception:
        pass

    x_vals = np.interp(y_grid, cys, cxs)
    return x_vals, y_grid, "numpy.interp"


def _fallback_pick_endplate(arr: np.ndarray, which: str) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback endplate selector using image Y ordering only."""
    if which == "superior":
        pts = arr[np.argsort(arr[:, 1])][:2]
    else:
        pts = arr[np.argsort(arr[:, 1])[::-1]][:2]
    p1, p2 = pts[0], pts[1]
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    return p1, p2


def _pick_endplate_from_reference_spline(
    arr: np.ndarray,
    x_vals: Optional[np.ndarray],
    y_vals: Optional[np.ndarray],
    which: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick superior/inferior endplate points using the local spline direction as reference."""
    if arr.shape[0] < 2:
        raise ValueError("Need at least 2 points to define an endplate")
    if arr.shape[0] == 2 or x_vals is None or y_vals is None or x_vals.shape[0] < 2 or y_vals.shape[0] < 2:
        return _fallback_pick_endplate(arr, which)

    centroid = arr.mean(axis=0)
    idx = int(np.argmin(np.abs(y_vals - float(centroid[1]))))
    dx = np.gradient(x_vals)
    dy = np.gradient(y_vals)
    tangent = np.array([float(dx[idx]), float(dy[idx])], dtype=float)
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1e-8:
        return _fallback_pick_endplate(arr, which)

    tangent /= tangent_norm
    if tangent[1] < 0:
        tangent *= -1.0

    endplate_axis = np.array([tangent[1], -tangent[0]], dtype=float)
    axis_norm = float(np.linalg.norm(endplate_axis))
    if axis_norm <= 1e-8:
        return _fallback_pick_endplate(arr, which)
    endplate_axis /= axis_norm
    if endplate_axis[0] < 0:
        endplate_axis *= -1.0

    tangent_proj = arr @ tangent
    if which == "superior":
        chosen = arr[np.argsort(tangent_proj)[:2]]
    else:
        chosen = arr[np.argsort(tangent_proj)[-2:]]

    axis_proj = chosen @ endplate_axis
    order = np.argsort(axis_proj)
    p1, p2 = chosen[order[0]], chosen[order[1]]
    if p1[0] > p2[0]:
        p1, p2 = p2, p1
    return p1, p2


def _build_payload_reference_spline(payload: Payload) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str]:
    """Build the global centroid-based spline reference used for endplate ordering."""
    return _build_reference_spline_xy(
        _compute_spline_vertebra_points(payload),
        smoothing_factor=_get_spline_smoothing_factor(payload),
    )


def _get_t1_y_threshold(payload: Payload) -> Optional[float]:
    """
    Return T1 centroid Y (image coordinates) when vertebra ordering allows it.

    Expected order after sorting by centroid Y descending (caudal -> cranial):
    [S3, S2,] S1(0), L5(1), ..., T12(6), ..., T1(17), C7(18), ...
    The sacral_extra offset accounts for extra sacral vertebrae (S2, S3).
    """
    sacral_extra = int(payload.sacral_extra or 0)
    group_centroids: List[Tuple[int, float]] = []
    for gi, group in enumerate(payload.points or []):
        if not group:
            continue
        garr = np.array([[float(p.x), float(p.y)] for p in group], dtype=float)
        if garr.shape[0] == 0:
            continue
        cy = float(garr[:, 1].mean())
        group_centroids.append((gi, cy))

    t1_index = 17 + sacral_extra
    if len(group_centroids) < t1_index + 1:
        return None

    group_centroids.sort(key=lambda t: t[1], reverse=True)
    return float(group_centroids[t1_index][1])


def _get_l1_y_threshold(payload: Payload) -> Optional[float]:
    """Return L1 centroid Y (image coordinates) when vertebra ordering allows it.
    Adjusts index by sacral_extra to handle S2/S3 being present.
    """
    sacral_extra = int(payload.sacral_extra or 0)
    group_centroids: List[Tuple[int, float]] = []
    for gi, group in enumerate(payload.points or []):
        if not group:
            continue
        garr = np.array([[float(p.x), float(p.y)] for p in group], dtype=float)
        if garr.shape[0] == 0:
            continue
        cy = float(garr[:, 1].mean())
        group_centroids.append((gi, cy))

    l1_index = 5 + sacral_extra
    if len(group_centroids) < l1_index + 1:
        return None

    group_centroids.sort(key=lambda t: t[1], reverse=True)
    return float(group_centroids[l1_index][1])


def _get_counted_centroid_xy(payload: Payload, index_from_s1: int) -> Optional[Tuple[float, float]]:
    """Return (x, y) of the vertebra at the given caudal->cranial index (S1=0), via automatic counting.
    Used to pick the inflection EUCLIDEAN-closest to a landmark (T1=17, L1=5), which is rotation-invariant
    (rigid rotation preserves distances) — unlike a 1D Y-distance, which flips between clustered candidates
    when the pelvis is rotated. Index is offset by sacral_extra (S2/S3)."""
    sacral_extra = int(payload.sacral_extra or 0)
    centroids: List[Tuple[float, float]] = []
    for group in payload.points or []:
        if not group:
            continue
        garr = np.array([[float(p.x), float(p.y)] for p in group], dtype=float)
        if garr.shape[0] == 0:
            continue
        centroids.append((float(garr[:, 0].mean()), float(garr[:, 1].mean())))
    idx = index_from_s1 + sacral_extra
    if len(centroids) < idx + 1:
        return None
    centroids.sort(key=lambda c: c[1], reverse=True)  # caudal first (largest y)
    return centroids[idx]


def _get_s1_superior_endplate_points(payload: Payload) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return S1 superior endplate endpoints (left, right) in image coordinates.
    Accounts for sacral_extra: when S2/S3 are present, S1 is no longer the most
    caudal group but sits at index sacral_extra in the caudal-sorted list.
    """
    sacral_extra = int(payload.sacral_extra or 0)
    sorted_groups: List[np.ndarray] = []
    for group in payload.points or []:
        if not group or len(group) < 2:
            continue
        garr = np.array([[float(p.x), float(p.y)] for p in group], dtype=float)
        if garr.shape[0] < 2:
            continue
        sorted_groups.append(garr)

    if not sorted_groups:
        return None

    # Sort caudal-first (largest Y first)
    sorted_groups.sort(key=lambda a: float(a[:, 1].mean()), reverse=True)

    s1_idx = min(sacral_extra, len(sorted_groups) - 1)
    best_arr = sorted_groups[s1_idx]

    if best_arr.shape[0] < 2:
        return None

    x_vals, y_vals, _ = _build_payload_reference_spline(payload)
    return _pick_endplate_from_reference_spline(best_arr, x_vals, y_vals, "superior")


def _sacral_endplate_tangent_angle(payload: Payload) -> Optional[float]:
    """Sacral-slope reference for the Degree-of-Curvature FIRST bar (L5-SS), expressed in the SAME
    tangent-angle convention as the per-vertebra centroid-slope angles (degrees(arctan2(dx, dy))).

    This is the NORMAL to the S1 superior endplate — the spine axis implied by the sacrum — so the
    first bar is measured against the TRUE sacral slope (consistent with SS/PI/PT/LL) instead of the
    unstable spline tangent at the S1 centroid used as a proxy. Same frame/sign as angles[0], so it is
    a drop-in reference. Returns None if the S1 endplate can't be determined (caller falls back).
    """
    eps = _get_s1_superior_endplate_points(payload)
    if eps is None:
        return None
    left, right = eps
    ex = float(right[0]) - float(left[0])   # endplate vector, left→right (ex >= 0 by construction)
    ey = float(right[1]) - float(left[1])
    nx, ny = -ey, ex                        # normal to the endplate (spine axis at S1)
    if ny < 0:                              # orient caudally (+y) to match the spline tangent
        nx, ny = -nx, -ny
    return float(np.degrees(np.arctan2(nx, ny)))


def _get_thoracolumbar_transition_y(payload: Payload) -> Optional[float]:
    """
    Return an estimated Y for the T12/L1 transition in image coordinates.

    Uses centroid ordering caudal->cranial (Y descending):
    [S3, S2,] S1(0+se), L5(1+se), L4(2+se), L3(3+se), L2(4+se), L1(5+se), T12(6+se).
    Adjusts indices by sacral_extra.
    """
    sacral_extra = int(payload.sacral_extra or 0)
    group_centroids: List[Tuple[int, float]] = []
    for gi, group in enumerate(payload.points or []):
        if not group:
            continue
        garr = np.array([[float(p.x), float(p.y)] for p in group], dtype=float)
        if garr.shape[0] == 0:
            continue
        cy = float(garr[:, 1].mean())
        group_centroids.append((gi, cy))

    l1_index = 5 + sacral_extra
    t12_index = 6 + sacral_extra
    if len(group_centroids) < t12_index + 1:
        return None

    group_centroids.sort(key=lambda t: t[1], reverse=True)
    l1_y = float(group_centroids[l1_index][1])
    t12_y = float(group_centroids[t12_index][1])
    return float((l1_y + t12_y) / 2.0)


def _get_anatomic_endplate_lines(payload: Payload) -> dict:
    """Return anatomical endplate lines as {label: (left, right)} in image coordinates.
    Adjusts indices by sacral_extra to handle S2/S3 being present.
    """
    sacral_extra = int(payload.sacral_extra or 0)
    group_arrays: List[np.ndarray] = []
    for group in payload.points or []:
        if not group or len(group) < 2:
            continue
        garr = np.array([[float(p.x), float(p.y)] for p in group], dtype=float)
        if garr.shape[0] < 2:
            continue
        group_arrays.append(garr)

    if not group_arrays:
        return {}

    group_arrays.sort(key=lambda arr: float(arr[:, 1].mean()), reverse=True)

    x_vals, y_vals, _ = _build_payload_reference_spline(payload)

    s1_idx  = sacral_extra
    l1_idx  = 5 + sacral_extra
    t12_idx = 6 + sacral_extra
    t1_idx  = 17 + sacral_extra

    lines: dict = {}
    if len(group_arrays) > s1_idx:
        lines["s1"] = _pick_endplate_from_reference_spline(group_arrays[s1_idx], x_vals, y_vals, "superior")
    if len(group_arrays) > l1_idx:
        lines["l1"] = _pick_endplate_from_reference_spline(group_arrays[l1_idx], x_vals, y_vals, "superior")
    if len(group_arrays) > t12_idx:
        lines["t12"] = _pick_endplate_from_reference_spline(group_arrays[t12_idx], x_vals, y_vals, "inferior")
    if len(group_arrays) > t1_idx:
        lines["t1"] = _pick_endplate_from_reference_spline(group_arrays[t1_idx], x_vals, y_vals, "superior")
    return lines


def _line_spline_intersection(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    line_left: np.ndarray,
    line_right: np.ndarray,
    target_y: float,
) -> Optional[Tuple[np.ndarray, int]]:
    """Return (intersection_point, nearest_index) for a spline polyline crossing an anatomical line."""
    if x_vals.shape[0] < 2 or y_vals.shape[0] < 2:
        return None

    pts = np.column_stack((x_vals, y_vals))
    line_vec = line_right - line_left
    line_norm = float(np.hypot(line_vec[0], line_vec[1]))
    if line_norm < 1e-8:
        return None

    signed = line_vec[0] * (pts[:, 1] - line_left[1]) - line_vec[1] * (pts[:, 0] - line_left[0])
    eps = 1e-8
    candidates: List[Tuple[np.ndarray, int, float]] = []

    for i in range(pts.shape[0] - 1):
        f0 = float(signed[i])
        f1 = float(signed[i + 1])
        hit = (abs(f0) <= eps) or (abs(f1) <= eps) or (f0 * f1 < 0.0)
        if not hit:
            continue

        if abs(f0) <= eps:
            p = pts[i]
            idx = i
        elif abs(f1) <= eps:
            p = pts[i + 1]
            idx = i + 1
        else:
            t = abs(f0) / (abs(f0) + abs(f1))
            p = pts[i] + t * (pts[i + 1] - pts[i])
            idx = int(round(i + t))

        idx = max(0, min(int(idx), int(pts.shape[0] - 1)))
        candidates.append((p, idx, abs(float(p[1]) - target_y)))

    if candidates:
        candidates.sort(key=lambda c: c[2])
        p, idx, _ = candidates[0]
        return p, idx

    # Fallback: choose spline point nearest to the line and target Y level.
    line_dist = np.abs(signed) / line_norm
    score = line_dist + 0.01 * np.abs(y_vals - target_y)
    idx = int(np.argmin(score))
    return pts[idx], idx


def _normal_theta_from_spline_gradient(dx: np.ndarray, dy: np.ndarray, idx: int) -> float:
    """Return undirected normal orientation in [0,180) from local spline gradient arrays."""
    safe_idx = max(0, min(int(idx), int(dx.shape[0] - 1)))
    denom = float(dy[safe_idx])
    slope = float(dx[safe_idx]) / denom if abs(denom) > 1e-12 else 0.0
    tangent_theta = float(np.degrees(np.arctan2(1.0, slope))) % 180.0
    return (tangent_theta + 90.0) % 180.0


# ================================ PELVIC PARAMETERS ================================
# Pure geometry, no FastAPI and no disk, so it can be published, cited and tested outside the server.
# Bad input raises ValueError; the HTTP layer turns that into a 400.
def compute_pelvic_parameters(payload: Payload) -> dict:
    """
    Compute pelvic parameters: Pelvic Incidence (PI), Pelvic Tilt (PT),
    Sacral Slope (SS), Lumbar Lordosis (LL, L1 superior endplate to S1 superior endplate),
    and Thoracic Kyphosis (TK, T1 superior endplate to T12 inferior endplate).

    Geometry (sagittal radiograph, image coordinates with Y pointing down):

    - **S1 identification**: The vertebra group whose centroid has the largest Y
      (most caudal in the image).
        - **S1 superior endplate**: The S1 edge selected from its marked vertices using
            the local spline direction as cranial-caudal reference.
    - **Hip Axis (HA)**: Midpoint of the two femoral head centres (or the single
      centre if only one head is marked).

    Definitions (Duval-Beaupère & Legaye):
      SS = angle between the S1 superior endplate and the horizontal.
      PT = angle between the vertical and the line from HA to the S1 endplate midpoint.
      PI = PT + SS  (always holds; also verified geometrically).
    """
    log("[PELVIC] Starting pelvic parameters calculation")

    # ---- 1. Identify S1 (most caudal vertebra group, adjusted for sacral_extra) ----
    if not payload.points or len(payload.points) < 1:
        raise ValueError("No vertebra groups provided")

    sacral_extra = int(payload.sacral_extra or 0)

    # Compute centroid Y for each group to find the caudal order
    group_centroids = []
    for gi, group in enumerate(payload.points):
        if not group or len(group) < 2:
            continue
        garr = np.array([[float(p.x), float(p.y)] for p in group], dtype=float)
        cy = float(garr[:, 1].mean())
        group_centroids.append((gi, cy, garr))

    if not group_centroids:
        raise ValueError("No valid vertebra groups found")

    # Sort caudal-first (largest Y first)
    group_centroids.sort(key=lambda t: t[1], reverse=True)

    # S1 is at index sacral_extra (0 in normal mode, 2 in sacral mode with S3+S2 above)
    s1_index = min(sacral_extra, len(group_centroids) - 1)
    s1_gi, s1_cy, s1_arr = group_centroids[s1_index]
    log(f"[PELVIC] S1 identified as group {s1_gi} (index={s1_index}, sacral_extra={sacral_extra}, centroid Y={s1_cy:.1f}, {s1_arr.shape[0]} points)")

    ref_x_vals, ref_y_vals, ref_method = _build_payload_reference_spline(payload)
    log(f"[PELVIC] Endplate ordering reference: {ref_method}")

    if s1_arr.shape[0] < 2:
        raise ValueError("S1 group needs at least 2 points for endplate detection")

    # ---- 2. S1 superior endplate guided by the local spline direction ----
    ep1, ep2 = _pick_endplate_from_reference_spline(s1_arr, ref_x_vals, ref_y_vals, "superior")

    s1_mid = (ep1 + ep2) / 2.0  # midpoint of superior endplate
    endplate_vec = ep2 - ep1     # vector along the endplate, left → right

    log(f"[PELVIC] S1 endplate: ({ep1[0]:.1f},{ep1[1]:.1f}) → ({ep2[0]:.1f},{ep2[1]:.1f})")
    log(f"[PELVIC] S1 midpoint: ({s1_mid[0]:.1f},{s1_mid[1]:.1f})")

    # ---- 3. Hip Axis (HA) ----
    ha_centers: List[np.ndarray] = []
    if payload.femoralHeads:
        for fh in payload.femoralHeads:
            if fh is not None and fh.center is not None:
                ha_centers.append(np.array([float(fh.center.x), float(fh.center.y)], dtype=float))

    if not ha_centers:
        raise ValueError("No femoral heads marked. Mark at least one femoral head.")

    hip_axis = np.mean(np.vstack(ha_centers), axis=0)
    log(f"[PELVIC] Hip axis: ({hip_axis[0]:.1f},{hip_axis[1]:.1f}) from {len(ha_centers)} head(s)")

    # ---- 3b. Identify posterosuperior corner of S1 ----
    # The femoral heads are ANTERIOR to the sacrum. Therefore the S1 endplate
    # point that is FARTHER from hip_axis (in X) is the posterior one.
    dist_ep1 = abs(float(ep1[0]) - float(hip_axis[0]))
    dist_ep2 = abs(float(ep2[0]) - float(hip_axis[0]))
    if dist_ep1 >= dist_ep2:
        s1_posterosuperior = ep1  # ep1 is farther from HA → posterior
    else:
        s1_posterosuperior = ep2  # ep2 is farther from HA → posterior
    log(f"[PELVIC] S1 posterosuperior corner: ({s1_posterosuperior[0]:.1f},{s1_posterosuperior[1]:.1f})"
        f" (dist_ep1={dist_ep1:.1f}, dist_ep2={dist_ep2:.1f})")

    # ---- 4. Compute SS (Sacral Slope) ----
    # Angle between the S1 endplate and the horizontal.
    # endplate_vec = (dx, dy); in image coords Y points down.
    # Horizontal = (1, 0). Angle = atan2(|dy|, |dx|).
    # We use the absolute angle because SS is always positive by convention.
    ep_dx = float(endplate_vec[0])
    ep_dy = float(endplate_vec[1])
    ss_rad = np.arctan2(abs(ep_dy), abs(ep_dx))
    SS = float(np.degrees(ss_rad))
    log(f"[PELVIC] SS = {SS:.2f}° (endplate_vec=({ep_dx:.1f},{ep_dy:.1f}))")

    # ---- 4b. Compute LL (Lumbar Lordosis L1-S1, static anatomical reference) ----
    # L1 is expected as the (6+sacral_extra)th vertebra when ordered caudal->cranial:
    # [S3, S2,] S1, L5, L4, L3, L2, L1.
    LL: Optional[float] = None
    l1_ep_left: Optional[np.ndarray] = None
    l1_ep_right: Optional[np.ndarray] = None
    l1_required = 6 + sacral_extra
    if len(group_centroids) >= l1_required:
        l1_gi, l1_cy, l1_arr = group_centroids[5 + sacral_extra]
        if l1_arr.shape[0] >= 2:
            l1_ep1, l1_ep2 = _pick_endplate_from_reference_spline(l1_arr, ref_x_vals, ref_y_vals, "superior")
            l1_ep_left = l1_ep1
            l1_ep_right = l1_ep2
            l1_vec = l1_ep2 - l1_ep1

            # Line orientation is undirected; normalize both to [0, 180)
            # and take the minimal inter-line angle.
            s1_theta = float(np.degrees(np.arctan2(ep_dy, ep_dx))) % 180.0
            l1_theta = float(np.degrees(np.arctan2(float(l1_vec[1]), float(l1_vec[0])))) % 180.0
            ll_diff = abs(s1_theta - l1_theta)
            if ll_diff > 90.0:
                ll_diff = 180.0 - ll_diff
            LL = float(ll_diff)
            log(
                f"[PELVIC] LL = {LL:.2f}° "
                f"(L1=group {l1_gi}, S1θ={s1_theta:.2f}°, L1θ={l1_theta:.2f}°)"
            )
        else:
            log(f"[PELVIC] LL unavailable: L1 group {l1_gi} has <2 points")
    else:
            log(
                f"[PELVIC] LL unavailable: fewer than {l1_required} vertebra groups (needs [S3/S2/]S1..L1)"
            )

    # ---- 4c. Compute TK (Thoracic Kyphosis T1-T12, static anatomical reference) ----
    # Expected indices in caudal->cranial order:
    # [S3(0), S2(1),] S1(0+se), L5(1+se), L4(2+se), L3(3+se), L2(4+se), L1(5+se), T12(6+se) ... T1(17+se)
    TK: Optional[float] = None
    t1_ep_left: Optional[np.ndarray] = None
    t1_ep_right: Optional[np.ndarray] = None
    t12_ep_left: Optional[np.ndarray] = None
    t12_ep_right: Optional[np.ndarray] = None
    t1_required = 18 + sacral_extra
    if len(group_centroids) >= t1_required:
        t12_gi, t12_cy, t12_arr = group_centroids[6 + sacral_extra]
        t1_gi, t1_cy, t1_arr = group_centroids[17 + sacral_extra]

        if t12_arr.shape[0] >= 2 and t1_arr.shape[0] >= 2:
            t12_ep1, t12_ep2 = _pick_endplate_from_reference_spline(t12_arr, ref_x_vals, ref_y_vals, "inferior")
            t12_ep_left = t12_ep1
            t12_ep_right = t12_ep2
            t12_vec = t12_ep2 - t12_ep1

            t1_ep1, t1_ep2 = _pick_endplate_from_reference_spline(t1_arr, ref_x_vals, ref_y_vals, "superior")
            t1_ep_left = t1_ep1
            t1_ep_right = t1_ep2
            t1_vec = t1_ep2 - t1_ep1

            t12_theta = float(np.degrees(np.arctan2(float(t12_vec[1]), float(t12_vec[0])))) % 180.0
            t1_theta = float(np.degrees(np.arctan2(float(t1_vec[1]), float(t1_vec[0])))) % 180.0
            tk_diff = abs(t12_theta - t1_theta)
            if tk_diff > 90.0:
                tk_diff = 180.0 - tk_diff
            TK = float(tk_diff)
            log(
                f"[PELVIC] TK = {TK:.2f}° "
                f"(T12=group {t12_gi}, T1=group {t1_gi}, T12θ={t12_theta:.2f}°, T1θ={t1_theta:.2f}°)"
            )
        else:
            log(
                f"[PELVIC] TK unavailable: insufficient points "
                f"(T12 group {t12_gi} pts={t12_arr.shape[0]}, T1 group {t1_gi} pts={t1_arr.shape[0]})"
            )
    else:
        log(f"[PELVIC] TK unavailable: fewer than {t1_required} vertebra groups (needs [S3/S2/]S1..T1)")

    # ---- 5. Compute PT (Pelvic Tilt) ----
    # Angle between the vertical and the line from HA to S1 endplate midpoint.
    # vec_ha_to_s1 = S1_mid - HA (points upward in the image because S1 is above HA)
    vec_ha_s1 = s1_mid - hip_axis  # (dx, dy) from HA to S1_mid
    vx = float(vec_ha_s1[0])
    vy = float(vec_ha_s1[1])
    vec_len = np.sqrt(vx ** 2 + vy ** 2)

    if vec_len < 1e-6:
        raise ValueError("S1 midpoint and hip axis overlap — cannot compute PT")

    # Vertical in image coords points down = (0, 1).
    # The line from HA to S1_mid generally points upward (vy < 0).
    # PT = angle between vec_ha_s1 and vertical_up (0, -1).
    # cos(PT) = dot(vec, (0,-1)) / |vec| = -vy / |vec|
    cos_pt = (-vy) / vec_len
    cos_pt = np.clip(cos_pt, -1.0, 1.0)  # numerical safety
    PT = float(np.degrees(np.arccos(cos_pt)))
    log(f"[PELVIC] PT = {PT:.2f}° (vec=({vx:.1f},{vy:.1f}), len={vec_len:.1f})")

    # ---- 6. Compute PI (Pelvic Incidence) ----
    # By definition: PI = PT + SS
    # Also verify geometrically: angle between the perpendicular to the endplate
    # at S1_mid and the line from S1_mid to HA.
    PI = PT + SS

    # Geometric verification
    # Perpendicular to endplate (pointing towards HA)
    perp1 = np.array([-ep_dy, ep_dx], dtype=float)   # rotate 90° CCW
    perp2 = np.array([ep_dy, -ep_dx], dtype=float)    # rotate 90° CW
    vec_s1_ha = hip_axis - s1_mid  # from S1_mid towards HA
    # Pick the perpendicular that points towards HA (positive dot product)
    perp = perp1 if np.dot(perp1, vec_s1_ha) > 0 else perp2
    # Angle between perp and vec_s1_ha
    dot_val = np.dot(perp, vec_s1_ha)
    cross_mag = np.sqrt(np.dot(perp, perp) * np.dot(vec_s1_ha, vec_s1_ha))
    if cross_mag > 1e-6:
        cos_pi = np.clip(dot_val / cross_mag, -1.0, 1.0)
        PI_geometric = float(np.degrees(np.arccos(cos_pi)))
    else:
        PI_geometric = PI

    log(f"[PELVIC] PI = {PI:.2f}° (PT+SS), geometric = {PI_geometric:.2f}°")

    notes_parts = [
        f"S1=group[{s1_gi}] ({s1_arr.shape[0]}pts)",
        f"HA from {len(ha_centers)} head(s)",
        f"PI geometric check: {PI_geometric:.1f}°",
        f"PosteroSup S1: ({s1_posterosuperior[0]:.1f},{s1_posterosuperior[1]:.1f})"
    ]
    if LL is not None:
        notes_parts.append(f"LL(L1-S1): {LL:.1f}°")
    else:
        notes_parts.append("LL(L1-S1): unavailable")
    if TK is not None:
        notes_parts.append(f"TK(T1-T12): {TK:.1f}°")
    else:
        notes_parts.append("TK(T1-T12): unavailable")

    return {
        "PI": PI, "PT": PT, "SS": SS, "LL": LL, "TK": TK,
        "PI_geometric": PI_geometric,
        "s1_endplate_midpoint": s1_mid,
        "s1_endplate_left": ep1,
        "s1_endplate_right": ep2,
        "s1_posterosuperior": s1_posterosuperior,
        "l1_endplate_left": l1_ep_left,
        "l1_endplate_right": l1_ep_right,
        "t1_endplate_left": t1_ep_left,
        "t1_endplate_right": t1_ep_right,
        "t12_endplate_left": t12_ep_left,
        "t12_endplate_right": t12_ep_right,
        "hip_axis": hip_axis,
        "notes": " | ".join(notes_parts),
    }


# ============================== CENTROID INCLINATION ==============================
def compute_vertebra_inclination_angles(payload: Payload) -> List[dict]:
    """
    Compute inclination angles between successive vertebral centroids.
    Angle is calculated relative to the vertical (Y-axis).
    For each pair of successive vertebrae, we compute the vector and its angle.
    """
    verts = _compute_vertebra_centroids(payload)

    if len(verts) < 2:
        return []

    # Sort by Y (top to bottom)
    sorted_verts = sorted([(i, v) for i, v in enumerate(verts)], key=lambda x: x[1].y)

    out: List[dict] = []

    # Calculate angle between each pair of successive vertebrae
    for idx in range(len(sorted_verts) - 1):
        i1, v1 = sorted_verts[idx]
        i2, v2 = sorted_verts[idx + 1]

        # Vector from v1 to v2
        dx = float(v2.x) - float(v1.x)
        dy = float(v2.y) - float(v1.y)

        # Angle relative to vertical (positive Y-axis points down in image coords)
        # atan2(dx, dy) gives angle from vertical axis
        angle_rad = np.arctan2(dx, dy) if dy != 0 else 0.0
        angle_deg = float(np.degrees(angle_rad))

        # Store angle for the second vertebra in the pair
        out.append({"vertebra_index": i2, "vertebra_label": f"V{i2 + 1}", "angle_degrees": angle_deg})

    return out


# ================================= CURVATURE BY LEVEL =================================
def compute_curvature(payload: Payload) -> dict:
    """
    Angle differences between successive vertebrae (angleDiff), CentroidSlope method of
    SpineSlope.py (lines 1070-1105): tangent of the reference spline at each centroid, then the
    successive differences, the first one referenced to the real S1 superior endplate.
    """
    verts = _compute_spline_vertebra_points(payload)
    
    if len(verts) < 2:
        raise ValueError("Insufficient vertebrae for chart generation")
    
    # First, compute the spline (same as /spline endpoint)
    cys, cxs = _prepare_xy_for_spline(verts)
    if cys.shape[0] < 2:
        raise ValueError("Insufficient unique Y coordinates for spline")
    
    # Build spline (matching SpineSlope.py lines 573-578)
    try:
        if _HAS_SCIPY and UnivariateSpline is not None:
            k = 3 if cys.shape[0] >= 4 else max(1, int(cys.shape[0] - 1))
            spl = UnivariateSpline(cys, cxs, k=k)
            spl.set_smoothing_factor(_get_spline_smoothing_factor(payload))
        else:
            raise RuntimeError("SciPy required for Degree of Curvature calculation")
    except Exception as e:
        raise RuntimeError(f"Error creating spline: {str(e)}")
    
    # Sample the spline to get points and calculate derivatives
    y_min = float(cys.min())
    y_max = float(cys.max())
    y_samples = np.linspace(y_min, y_max, 1000)
    x_samples = spl(y_samples)
    
    # Calculate the derivative (tangent) at each point
    dx_dy = spl.derivative()(y_samples)
    
    # Calculate angles using CentroidSlope method:
    # For each centroid, find closest point on spline and get its tangent angle
    angles: List[float] = []
    
    # Sort vertebrae by Y (bottom to top in spine: S1/sacro first, C2/cervical last)
    # In image coordinates: larger Y = bottom of image = caudal (sacro)
    # So we sort by Y DESCENDING to match SpineSlope.py order: S1, L5, L4, ..., C2
    sorted_verts = sorted([(i, v) for i, v in enumerate(verts)], key=lambda x: x[1].y, reverse=True)
    
    for i, (idx, vert) in enumerate(sorted_verts):
        # Find closest point on spline to this centroid
        cy = float(vert.y)
        cx = float(vert.x)
        
        # Find index of closest Y value in spline samples
        closest_idx = np.argmin(np.abs(y_samples - cy))
        
        # Get the tangent angle at this point
        slope = dx_dy[closest_idx]
        angle_rad = np.arctan2(slope, 1.0)  # arctan2(dx, dy) with dy=1
        angle_deg = float(np.degrees(angle_rad))
        angles.append(angle_deg)
    
    # Calculate angleDiff (differences between successive angles)
    # Matching SpineSlope.py lines 1073-1095 exactly
    angle_diffs: List[float] = []
    
    # First difference: angle[1] − sacral-slope reference (the L5-SS bar).
    # SpineSlope.py used the REAL S1 superior endplate: angleDiff.append(-1*(angle[1]- Angle(vS1[3],vS1[2])...)).
    # We now do the same — reference the true sacral slope (S1 endplate normal, same tangent convention as
    # angles[]), consistent with SS/PI/PT/LL — instead of angles[0] (the spline tangent at the S1 centroid),
    # which is an unstable proxy. Falls back to angles[0] only if the S1 endplate can't be determined.
    if len(angles) >= 2:
        ss_ref = _sacral_endplate_tangent_angle(payload)
        sacrum_angle = ss_ref if ss_ref is not None else angles[0]
        angle_diffs.append(-1 * (angles[1] - sacrum_angle))
    
    # Remaining differences (matching SpineSlope.py line 1074-1093)
    for idx in range(1, len(angles) - 1):
        diff = -1 * (angles[idx + 1] - angles[idx])
        angle_diffs.append(diff)
    
    # Handle last angle (C2) - SpineSlope.py lines 1094-1097 sets it to 0 if positive
    if len(angles) >= 2:
        dc2 = -1 * (angles[-1] - angles[-2])
        if dc2 > 0:
            dc2 = 0
        angle_diffs.append(dc2)
    
    return {"angles": angles, "angle_diffs": angle_diffs}
