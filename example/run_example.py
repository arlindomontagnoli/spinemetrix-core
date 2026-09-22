"""Computes the sagittal parameters of the example case and prints them."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from spinemetrix_core import compute_pelvic_parameters, load_payload

payload = load_payload(json.load(open(os.path.join(HERE, "marks.json"))))
r = compute_pelvic_parameters(payload)

print(f"Marked vertebrae: {len(payload.points)}")
for k in ("PI", "PT", "SS", "LL", "TK"):
    v = r[k]
    print(f"{k:>3} = {v:6.2f} deg" if v is not None else f"{k:>3} = unavailable")
print(f"\nGeometric PI (independent check of PT+SS): {r['PI_geometric']:.2f}")
print(f"Hip axis: ({r['hip_axis'][0]:.1f}, {r['hip_axis'][1]:.1f})")
