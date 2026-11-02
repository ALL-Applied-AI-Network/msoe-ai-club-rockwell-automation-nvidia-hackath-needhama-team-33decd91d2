"""Read-only measurement of the supplied Dataset 1 stage. Run through Slurm."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    try:
        from pxr import Usd, UsdGeom
        root = Path(__file__).resolve().parents[2]
        source = root / 'shared/data/csi_digital_twin_v1/stations/tray_inspection/tray_inspection.usdc'
        stage = Usd.Stage.Open(str(source))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])
        report = {'source': str(source), 'meters_per_unit': UsdGeom.GetStageMetersPerUnit(stage), 'prims': {}}
        for prim in stage.GetPrimAtPath('/TrayInspection').GetChildren():
            name = prim.GetName()
            if name.startswith(('G', 'tray_good')):
                xf = UsdGeom.Xformable(prim)
                matrix = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                report['prims'][name] = {
                    'origin_world_m': list(matrix.ExtractTranslation()),
                    'bounds_min_m': list(bounds.GetMin()), 'bounds_max_m': list(bounds.GetMax()),
                    'transform': [list(row) for row in matrix],
                    'type': prim.GetTypeName(),
                }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + '\n')
        print('SOURCE_GEOMETRY_REPORT', out)
    finally:
        app.close()


if __name__ == '__main__':
    main()
