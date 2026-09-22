"""Checks that the core reproduces the reference values of the example case.

Runs without pytest:  python3 tests/test_core.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import spinemetrix_core as core

EX = os.path.join(os.path.dirname(HERE), "example")


def test_pelvic_parameters_match_reference():
    payload = core.load_payload(json.load(open(os.path.join(EX, "marks.json"))))
    expected = json.load(open(os.path.join(EX, "expected.json")))["pelvic"]
    got = core.compute_pelvic_parameters(payload)

    for key, ref in expected.items():
        if key == "points":
            continue
        assert round(got[key], 2) == ref, f"{key}: {round(got[key], 2)} != {ref}"

    for key, ref in expected["points"].items():
        p = got[key]
        if ref is None:
            assert p is None, f"{key} should be None"
            continue
        assert abs(float(p[0]) - ref[0]) < 1e-6 and abs(float(p[1]) - ref[1]) < 1e-6, key


def test_curvature_matches_reference():
    payload = core.load_payload(json.load(open(os.path.join(EX, "marks.json"))))
    expected = json.load(open(os.path.join(EX, "expected.json")))["curvature"]["angle_diffs"]
    got = [round(float(d), 4) for d in core.compute_curvature(payload)["angle_diffs"]]
    assert got == expected, f"curvature: {got} != {expected}"


def test_inclination_matches_reference():
    payload = core.load_payload(json.load(open(os.path.join(EX, "marks.json"))))
    expected = json.load(open(os.path.join(EX, "expected.json")))["inclination"]
    got = core.compute_vertebra_inclination_angles(payload)
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert g["vertebra_index"] == e["vertebra_index"]
        assert abs(g["angle_degrees"] - e["angle_degrees"]) < 1e-6


if __name__ == "__main__":
    test_pelvic_parameters_match_reference()
    test_curvature_matches_reference()
    test_inclination_matches_reference()
    print("ok - parameters, curvature and inclination match the reference")
