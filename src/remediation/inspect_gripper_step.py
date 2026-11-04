"""Record reproducible, kernel-free facts from the supplied custom gripper STEP file.

This intentionally does not claim that raw STEP coordinates define the robot TCP or
contact surfaces.  Those require a CAD import, mounting-pose decision, and contact
validation in Isaac Sim.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POINT = re.compile(r"CARTESIAN_POINT\s*\([^,]*,\s*\(([^)]*)\)\)", re.I)


def inspect(path: Path):
    raw = path.read_bytes()
    text = raw.decode('utf-8', errors='replace')
    points = []
    for match in POINT.finditer(text):
        values = [float(value.strip()) for value in match.group(1).split(',')]
        if len(values) == 3:
            points.append(values)
    if not points:
        raise ValueError('No CARTESIAN_POINT records found')
    axes = ('x', 'y', 'z')
    try:
        display_path = str(path.relative_to(ROOT))
    except ValueError:
        display_path = str(path)
    return {
        'path': display_path,
        'sha256': hashlib.sha256(raw).hexdigest(),
        'schema': 'AP242' if 'AP242' in text.upper() else 'unknown',
        'metre_unit_declared': '.METRE.' in text.upper(),
        'manifold_solid_brep_count': len(re.findall(r'MANIFOLD_SOLID_BREP', text, re.I)),
        'cartesian_point_count': len(points),
        'raw_cartesian_envelope_m': {
            axis: [min(point[i] for point in points), max(point[i] for point in points)]
            for i, axis in enumerate(axes)
        },
        'limitations': 'Raw STEP bounds are not a robot mounting pose, TCP, or collision/contact validation.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=Path, default=ROOT / 'data/gripper.step')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    report = inspect(args.step)
    encoded = json.dumps(report, indent=2) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded)
    else:
        print(encoded, end='')
