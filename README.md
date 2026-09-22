# SpineMetrix core

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22900527.svg)](https://doi.org/10.5281/zenodo.22900527)

Open-source implementation of the sagittal spinal parameters computed by
[SpineMetrix](https://spinemetrix.com) — pelvic incidence (PI), pelvic tilt (PT), sacral slope (SS),
lumbar lordosis (LL) and thoracic kyphosis (TK) — published so that results obtained with the
application can be inspected, verified and reproduced independently.

This repository holds the **measurement code only**. It is the same code that runs in production: no
interface, no database, no patient data. Marked vertebra corners go in, parameters come out.

## Why this exists

A number in a paper is only checkable if the reader can see how it was computed. The formulas below
are standard, but every implementation makes choices — which endplate is "superior", how the spline
is smoothed, how the posterior corner is identified — and those choices move the numbers. They are
all visible here.

## Install and run

```
pip install numpy scipy      # scipy is optional; without it the spline falls back to linear
python3 example/run_example.py
python3 tests/test_core.py
```

The example uses `example/marks.json`, a 24-vertebra study with no patient data.
`example/expected.json` holds the reference values produced by the production server, and the test
fails if this code ever drifts from them.

## Input

A JSON as saved by SpineMetrix, or any object with the same shape:

```json
{"payload": {
  "points": [[{"x": 1200, "y": 6100}, {"x": 1480, "y": 6090}, ...], ...],
  "femoralHeads": [{"center": {"x": 1339, "y": 6355}, "radius": 180}],
  "sacral_extra": 0,
  "image_height": 2000
}}
```

`points` is one group of 4 corners per vertebra, in image pixels, **Y growing downwards**. Groups may
come in any order: they are sorted internally by centroid Y, most caudal first. `sacral_extra` is 2
when S3 and S2 were also marked above S1. `femoralHeads` may hold one or two circles; the hip axis is
their midpoint.

## What is computed

`compute_pelvic_parameters(payload)` returns the angles in degrees plus every landmark used to get
them, so each step can be drawn and checked:

- **SS** — angle between the S1 superior endplate and the horizontal.
- **PT** — angle between the vertical and the line from the hip axis to the S1 endplate midpoint.
- **PI** — PT + SS, following Duval-Beaupère and Legaye. The returned `PI_geometric` recomputes it
  independently, as the angle between the perpendicular to the S1 endplate and the line to the hip
  axis; the two agreeing is a check that the landmarks are consistent.
- **LL** — minimal angle between the L1 and S1 superior endplates.
- **TK** — minimal angle between the T1 superior and T12 inferior endplates.

`compute_curvature(payload)` returns the degree of curvature per level: the tangent of the reference
spline at each vertebra centroid, and the successive differences between them. The first difference
is referenced to the real S1 superior endplate rather than to the spline tangent at the S1 centroid,
which is unstable. `compute_vertebra_inclination_angles(payload)` returns the inclination of each
segment between successive centroids, relative to the vertical.

Vertebra levels are not detected from the image: they are counted from S1 upwards, which is why the
marking protocol requires S1 and C2 to be visible and marked.

Two implementation details worth knowing, because they are the ones that change results:

**Endplate selection.** Which two of the four marked corners form an endplate is decided by the local
direction of a reference spline through the vertebra centroids, not by sorting corners by Y. Sorting
by Y breaks on a strongly tilted vertebra, such as S1 or the apex of a large curve.

**Spline smoothing.** The reference spline is a `scipy` `UnivariateSpline` of x as a function of y,
with smoothing factor `s = H / 80`, where `H` is the image height in pixels (20 when the height is
unknown). Scaling with image height keeps the smoothing equivalent across different resolutions.

## Scope

Present: the pelvic and regional parameters listed above, the reference spline, and the endplate
geometry they depend on.

Not here yet: Cobb from inflection points, Roussouly classification, T4-L1-Hip,
and the surgical planning model. They are being extracted from the application in the same way, one
verified step at a time, and will land here.

## Citing

Cite the archived release with its DOI rather than this repository directly, so the version is
pinned: [https://doi.org/10.5281/zenodo.22900527](https://doi.org/10.5281/zenodo.22900527). See `CITATION.cff` for the full entry.

## License

Apache-2.0 — see `LICENSE`. This is research and educational software; it is **not a medical
device** and must not be the sole basis for any clinical decision.
