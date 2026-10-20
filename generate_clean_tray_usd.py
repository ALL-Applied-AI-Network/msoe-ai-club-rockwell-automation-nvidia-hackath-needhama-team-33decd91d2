from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})
import omni.usd
from pxr import Usd, UsdGeom, Gf, Sdf
from pathlib import Path

src_stage_path = 'shared/data/csi_digital_twin_v1/stations/tray_inspection/tray_inspection.usdc'
src_stage = Usd.Stage.Open(src_stage_path)

out_usd_path = Path('shared/scenes/clean_vial_tray.usda')
out_usd_path.parent.mkdir(parents=True, exist_ok=True)
if out_usd_path.exists():
    out_usd_path.unlink()

out_stage = Usd.Stage.CreateNew(str(out_usd_path))
UsdGeom.SetStageUpAxis(out_stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(out_stage, 1.0)

# The /TrayInspection prim in tray_inspection.usdc has scale (0.0254, 0.0254, 0.0254) (inches to meters)
# Inside /TrayInspection, all children coordinates are in INCHES!
# tray_good_top was at (-14.1122, -25.2324, 46.8831) inches.
# Let's define the origin of the tray at its geometric center (X_mid, Y_mid) and base (Z_base):
# X: between G01 (-19.11 in) and G30 (-9.08 in) -> center ~ -14.10 in
# Y: between G01 (-30.05 in) and G05 (-20.40 in) -> center ~ -25.23 in
# Z_base: bottom of tray plate ~ 46.88 in (1.1908 m).
# Vial caps sit at 48.848 in (which is +1.965 in = +0.050 m above tray_good_top).

origin_in_inches = Gf.Vec3d(-14.10, -25.23, 46.88)
scale_m_per_in = 0.0254

tray_root = UsdGeom.Xform.Define(out_stage, "/CleanTray")
out_stage.SetDefaultPrim(tray_root.GetPrim())

good_prims = [
    'tray_good_top', 'tray_good_support', 
    'tray_good_post', 'tray_good_post_001', 'tray_good_post_002', 'tray_good_post_003',
    'note_good', 'tray_label', 'tray_label_001', 'tray_label_002', 'tray_label_003'
]
for i in range(1, 31):
    good_prims.append(f"G{i:02d}_tube")
    good_prims.append(f"G{i:02d}_cap")

for name in good_prims:
    p = src_stage.GetPrimAtPath(f"/TrayInspection/{name}")
    if p.IsValid():
        Sdf.CopySpec(src_stage.GetRootLayer(), p.GetPath(), out_stage.GetRootLayer(), Sdf.Path(f"/CleanTray/{name}"))
        dst_prim = out_stage.GetPrimAtPath(f"/CleanTray/{name}")
        xf = UsdGeom.Xformable(dst_prim)
        op = next((o for o in xf.GetOrderedXformOps() if "translate" in o.GetOpName()), None)
        if op:
            orig = op.Get()
            # Convert inch coordinates to meters relative to tray origin
            dx_m = (orig[0] - origin_in_inches[0]) * scale_m_per_in
            dy_m = (orig[1] - origin_in_inches[1]) * scale_m_per_in
            dz_m = (orig[2] - origin_in_inches[2]) * scale_m_per_in
            op.Set(Gf.Vec3d(dx_m, dy_m, dz_m))
        
        # Scale the geometry from inches to meters
        scale_op = next((o for o in xf.GetOrderedXformOps() if "scale" in o.GetOpName()), None)
        if scale_op:
            s = scale_op.Get()
            scale_op.Set(Gf.Vec3f(s[0] * scale_m_per_in, s[1] * scale_m_per_in, s[2] * scale_m_per_in))
        else:
            xf.AddScaleOp().Set(Gf.Vec3f(scale_m_per_in, scale_m_per_in, scale_m_per_in))

# Also copy materials
mat_scope = src_stage.GetPrimAtPath("/TrayInspection/_materials")
if mat_scope.IsValid():
    Sdf.CopySpec(src_stage.GetRootLayer(), mat_scope.GetPath(), out_stage.GetRootLayer(), Sdf.Path("/CleanTray/_materials"))

out_stage.GetRootLayer().Save()
print(f"Generated clean, isolated, metric tray USD at {out_usd_path}")
sim.close()
