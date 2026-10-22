import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.remediation.geometry import SPEC, base_to_uf0, cap_vertices, load_layout, rotation, spawn_height, transit_base_z, rim_base_z, uf0_to_base
from src.inspection.vision_pipeline import pixel_to_base, load_cameras
from src.remediation.validate_tablecam_tcp import validate as validate_tablecam_tcp


class GeometryTests(unittest.TestCase):
    def test_spawn_clearance_at_all_orientations(self):
        for tilt in range(0,181,5):
            for yaw in range(0,360,45):
                R=rotation(tilt,yaw)
                z=spawn_height(SPEC.surface_world_z,R)
                low=(cap_vertices()@R.T)[:,2].min()+z
                self.assertAlmostEqual(low-SPEC.surface_world_z,SPEC.spawn_clearance,places=10)

    def test_frame_roundtrip_and_table_uf0(self):
        table=np.array([.4,-.2,SPEC.surface_base_z])
        self.assertAlmostEqual(base_to_uf0(table)[2],-335.1)
        np.testing.assert_allclose(uf0_to_base(base_to_uf0(table)),table)

    def test_resized_camera_projection(self):
        cam=load_cameras()['table']; p=np.array([.385,.015,rim_base_z()])
        pc=np.linalg.inv(cam['T'])@np.r_[p,1]
        xy=pc[:2]/pc[2]; xy*=1+cam['k1']*np.dot(xy,xy)
        uv=xy*cam['fx']+np.array(cam['size'])/2
        for scale in [1,2,3]:
            np.testing.assert_allclose(pixel_to_base(cam,uv/scale,p[2],tuple(x/scale for x in cam['size'])),p,atol=1e-9)

    def test_unknown_crop_is_rejected(self):
        with self.assertRaises(ValueError): pixel_to_base(load_cameras()['table'],[320,240],image_size=(640,480))

    def test_tray_and_carried_cap_clearance(self):
        self.assertEqual(len(load_layout()['slots_m']),30)
        self.assertGreater(transit_base_z()-SPEC.plug_depth,rim_base_z()+.025)
        self.assertGreaterEqual(transit_base_z()-SPEC.surface_base_z,.26)

    def test_measured_tablecam_reference_validator(self):
        cam = load_cameras()['table']
        points = [np.array([x, y, rim_base_z()])
                  for x, y in ((.30, -.14), (.45, -.14), (.30, .02),
                               (.45, .02), (.375, -.06), (.42, -.10))]
        correspondences = []
        for i, point in enumerate(points):
            pc = np.linalg.inv(cam['T']) @ np.r_[point, 1]
            xy = pc[:2] / pc[2]; xy *= 1 + cam['k1'] * np.dot(xy, xy)
            correspondences.append({'id': str(i), 'pixel_xy': (xy * cam['fx'] + np.array(cam['size']) / 2).tolist(),
                                    'feature_plane_base_z_m': point[2],
                                    'measured_feature_base_pos_m': point.tolist()})
        result = validate_tablecam_tcp(correspondences, cam)
        self.assertTrue(result['contact_localization_validated'])


if __name__=='__main__': unittest.main()
