from isaacsim import SimulationApp
sim = SimulationApp({"headless": True})
from pxr import Usd, UsdGeom
stage = Usd.Stage.Open('runs/isolated_tray.usda')
for p in ['/Tray/tray_good_top', '/Tray/G01_tube', '/Tray/G01_cap', '/Tray/G05_cap']:
    prim = stage.GetPrimAtPath(p)
    if prim.IsValid():
        xf = UsdGeom.Xformable(prim)
        print(p, xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
sim.close()
