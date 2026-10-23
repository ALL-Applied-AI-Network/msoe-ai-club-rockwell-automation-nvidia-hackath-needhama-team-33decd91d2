# Physical workcell measurement checklist

Use this checklist at the physical robot before enabling manipulation. Record
each new value with units, the measuring tool, date, operator, and a photo or
raw capture path. A value labelled **design value** comes from the repository
or CAD/simulation; it is not a physical measurement. Leave the recorded-value
column blank until it is measured on the actual workcell.

## How to use this checklist

For a position, write three numbers: **X, Y, Z**, measured in the robot base
frame. X is forward from the robot, Y is left from the robot, and Z is upward.
For an angle, write degrees. For a distance, write millimetres unless the row
explicitly requests metres or pixels. A photograph alone is supporting evidence;
it does not replace the number.

The **Current value** column has one of these meanings:

| Label in the current-value column | What it means |
| --- | --- |
| Design value | The intended dimension used to build the simulator. Measure the real part before relying on it. |
| Dataset-reference value | A value recovered from Dataset 1. It describes the reference asset, not necessarily the physical tray. |
| Simulation value | A value used in Isaac Sim so a rendered scene works. It is not a physical calibration. |
| Existing cell calibration | A value shipped with the hackathon cell software. Verify it on the installed camera/tool setup. |
| No physical value/result | Nothing has been measured or validated for this project yet. |

## Names used in this checklist

| Plain name | Meaning |
| --- | --- |
| Robot base frame | The robot's fixed coordinate system. X points forward, Y points left, and Z points upward. Positions in this frame use metres unless stated otherwise. |
| Tool center point | The one physical point on the gripper that a robot command treats as the end of the tool. For this task, record separate points for cap pickup, cap placement, and cap pressing if they differ. |
| Table camera | The fixed overhead camera viewing the tray and replacement-cap supply. |
| Wrist camera | The moving camera mounted near the robot wrist. Its position changes with the robot joints. |
| Camera calibration | The relationship between camera pixels, camera depth, and positions in the robot base frame. |
| Fiducial | A deliberately placed visual marker with a known identity and measured position, used to find the tray pose reliably. |
| Aperture | One circular hole in the silver tray plate that a vial passes through. |
| Colour/depth alignment | Whether a pixel in the colour image and the corresponding depth reading describe the same real-world point. |
| Native image | An image saved at the camera’s original resolution, with no crop, resize, rotation, or screenshot scaling. |
| Image pixel | A location in an image, written as horizontal pixel position then vertical pixel position. |
| Yaw | The tray’s rotation when viewed from above; zero means its reference edge is aligned with the robot-base axes. |
| Vial rim | The top circular lip of a vial on which a seated cap flange rests. |
| Cap flange | The wide, flat circular part at the top of a cap. |
| Cap plug | The narrower part below the flange that enters the vial mouth. |
| Seat | The final fully inserted cap position, with the flange resting on the vial rim. |
| Current value | A documented design, CAD, simulator, or existing calibration value. It is not automatically verified on the real cell. |

## Exact gripper reference points

Record these as physical points, even if they later prove to be the same point:

| Reference point | Exact point to touch/measure | Why it is separate |
| --- | --- | --- |
| Pickup point | The center of the two finger grooves when they have closed around the middle of a loose cap. | This determines where the robot must place the gripper to capture a replacement cap. |
| Placement point | The point on the carried cap’s center axis that must align with the center of an empty vial mouth before releasing. | The cap may sit below or ahead of the groove center while carried. |
| Press point | The center of the lowest flat surface intended to push a crooked cap downward. | This is the contact point used to reseat a crooked cap and may differ from the pickup point. |

## 1. Coordinate system and cameras

| Done | Measurement to record | How to measure it | Current value | Recorded physical value |
| --- | --- | --- | --- | --- |
| [ ] | Robot-base work-surface height | Touch the bare work surface with a calibrated probe/tool and record its Z position in the robot base frame. | −5.1 mm in the existing cell calibration | |
| [ ] | Table-camera colour image size | Save an untouched native camera image and record width × height in pixels. | 1920 × 1080 pixels | |
| [ ] | Table-camera depth image size | Save the matching raw depth image and record width × height in pixels. | 640 × 400 pixels | |
| [ ] | Table-camera colour/depth alignment | Capture a planar checkerboard or fiducial at several tray locations. Confirm the same physical point has matching colour and depth coordinates. Record maximum error in millimetres. | Existing calibration files only; no current field validation result | |
| [ ] | Table-camera position calibration | For at least 6 reference points spread over at least 120 mm in both table directions, record: native image pixel, actual point height, and measured robot-base X/Y/Z. | Capture form requires 6 points and 120 mm span | |
| [ ] | Table-camera accuracy | After calibration, test at least 10 separate points across tray and supply areas. Record median and maximum position error in millimetres. | No physical result | |
| [ ] | Wrist-camera colour image size | Save a native wrist-camera image. | 1280 × 720 pixels delivered; capture may be 848 × 480 pixels | |
| [ ] | Wrist-camera depth image size | Save the matching raw wrist depth image. | 848 × 480 pixels | |
| [ ] | Wrist-camera-to-gripper calibration | Hold the gripper at several robot poses, observe one fixed fiducial, and compare its wrist-camera position with its measured robot-base position. Record error. | Existing calibration file only; no current field validation result | |
| [ ] | Camera latency | Record elapsed time from a visible motion to the image/depth update for table and wrist cameras. | No current value | |

The six-point TableCam form is [tablecam_tcp_capture_template.json](/home/ad.msoe.edu/needhama/Documents/GitHub/innovation-lab-fall-2026/src/remediation/config/tablecam_tcp_capture_template.json). Save native images without cropping or resizing.

## 2. Tray and vial fixture

| Done | Measurement to record | How to measure it | Current value | Recorded physical value |
| --- | --- | --- | --- | --- |
| [ ] | Tray outer length and width | Measure the black base outer edges with calipers or a steel rule. | Dataset-reference top plate is 312.88 mm × 312.88 mm; not a physical fixture measurement | |
| [ ] | Black base thickness | Measure at an exposed edge. | Design value: 10.00 mm | |
| [ ] | Silver plate thickness | Measure at an exposed edge. | Design value: 3.00 mm | |
| [ ] | Height from black-base top to silver-plate underside | Measure beside a standoff. | Design value: 80.00 mm | |
| [ ] | Silver-plate top height above work surface | Measure from bare table to top surface. | Derived design value: 93.00 mm | |
| [ ] | Vial bottom support height | Measure from work surface to where each vial bottom rests. | Design value: black-base top, 10.00 mm above work surface | |
| [ ] | Vial rim height above work surface | Measure several vials seated in different tray locations. Record mean and range. | Derived design value: 213.20 mm | |
| [ ] | Aperture diameter | Measure at several holes in the silver plate. | Design value: 31.75 mm | |
| [ ] | Aperture center spacing, row direction | Measure centers of at least first, middle, and last columns. | Dataset 1 supplies only a reference pattern; physical pitch is not verified | |
| [ ] | Aperture center spacing, column direction | Measure centers of at least first, middle, and last rows. | Dataset 1 supplies only a reference pattern; physical pitch is not verified | |
| [ ] | Aperture circularity and burr clearance | Check each representative hole with a vial and cap flange; note burrs, snagging, or visible deformation. | No current physical result | |
| [ ] | Tray locating features | Measure pin, stop, nest, or edge-stop locations relative to robot base and identify the repeatable datum surfaces. | No current value | |
| [ ] | Tray placement repeatability | Remove and replace tray at least 10 times. Detect fiducials or measure the datum; record X/Y shift and yaw rotation range. | No current value | |
| [ ] | Tray fiducials | Record each marker’s identifier, size, and measured position relative to tray edges and robot base. | No current value | |

## 3. Vials and caps

| Done | Measurement to record | How to measure it | Current value | Recorded physical value |
| --- | --- | --- | --- | --- |
| [ ] | Vial outer diameter | Measure at several vials near rim and body. | Design value: 26.67 mm | |
| [ ] | Vial inner diameter | Measure inside several vial mouths with bore gauge/calipers. | Design value: 25.40 mm | |
| [ ] | Vial total height | Measure from bottom to rim. | Design value: 203.20 mm | |
| [ ] | Vial rim geometry | Measure rim thickness, flat landing width, and any chamfer. Photograph a cross-section if possible. | No current value | |
| [ ] | Cap flange outer diameter | Measure several caps at multiple angles. | Design value: 31.50 mm | |
| [ ] | Cap flange thickness | Measure with calipers. | Design value: 2.032 mm | |
| [ ] | Cap plug lower outer diameter | Measure near the closed bottom. | Design value: 22.86 mm | |
| [ ] | Cap plug upper outer diameter | Measure immediately beneath flange. | Approximate design range: 24.89–25.40 mm | |
| [ ] | Cap plug depth | Measure flange underside to plug bottom. | Design value: 13.46 mm | |
| [ ] | Cap wall thickness | Cut a sacrificial cap or use a suitable thickness gauge. Record plug wall and bottom-face thickness separately. | Design value: 2.00 mm wall and 2.00 mm flat bottom face | |
| [ ] | Cap seating travel | Measure distance from first plug entry to fully seated flange contact. | No current physical result | |
| [ ] | Crooked-cap stable condition | Create at least 10 intentionally crooked caps. Record tilt angle, visible rim contact, and whether each remains stable for 10 seconds. | Simulation reports about 40–44 degrees; no physical result | |
| [ ] | Press force/travel limit | Under supervision, record the minimum controlled motion/force that seats a crooked cap and the condition that indicates a stop. | No current value; do not assume force sensing is available | |

## 4. Custom fingers and robot tool geometry

| Done | Measurement to record | How to measure it | Current value | Recorded physical value |
| --- | --- | --- | --- | --- |
| [ ] | Finger mounting-hole center spacing | Measure the two holes on each finger and the matching moving jaw carrier holes. | CAD exists; no physical measurement | |
| [ ] | Mounting-face position | Measure each custom finger’s mounting face relative to the jaw carrier face after assembly. | Simulation visual mount: 18.5 mm forward and 6.5 mm downward from its baseline carrier reference | |
| [ ] | Finger minimum inside gap | Close the physical jaws slowly and measure nearest opposing finger surfaces. | Simulation mesh gap: 3 mm | |
| [ ] | Finger maximum inside gap | Open physical jaws fully and measure nearest opposing finger surfaces. | Simulation mesh gap: 15 mm | |
| [ ] | Groove width, depth, and radius | Measure every feature that contacts a cap, vial, or capped vial. | No physical measurement | |
| [ ] | Clearance to neighboring vials | Place gripper around representative vial/cap positions and measure clearance to adjacent vial and plate. | No current value | |
| [ ] | Gripper opening repeatability | Repeat fully open/closed positions at least 10 times and record gap variation. | No current value | |
| [ ] | Pickup tool center point | Touch a known reference with the physical cap-pick contact center. Record robot-base X/Y/Z and gripper state. Repeat at 6 or more poses. | No physical value | |
| [ ] | Placement tool center point | Touch a known vial-center/rim reference with the cap-placement point. Record robot-base X/Y/Z and gripper state. | No physical value | |
| [ ] | Press tool center point | Touch a known cap-top reference with the press face center. Record robot-base X/Y/Z and gripper state. | No physical value | |
| [ ] | Flange-to-tool distances | Measure wrist flange face to pickup, placement, and press tool centers along and across the tool. | Visual CAD envelope: 284.30 mm; provisional simulation route value: 100.00 mm; neither is a physical calibration | |
| [ ] | Tool orientation when pointing down | Record robot joint angles and tool orientation that put the gripper straight down over a known reference. | No physical value | |
| [ ] | Gripper command mapping | Verify which command is physically open and closed and measure the actual response time. | Simulation uses open/closed states only; no physical mapping result | |

## 5. Replacement-cap supply

| Done | Measurement to record | How to measure it | Current value | Recorded physical value |
| --- | --- | --- | --- | --- |
| [ ] | Supply fixture type and capacity | Photograph the fixture and record usable cap count. | No physical supply fixture defined | |
| [ ] | Pocket center positions | Record each pocket center X/Y/Z in robot-base frame or relative to a tray fiducial. | No physical value | |
| [ ] | Pocket spacing and wall height | Measure cap-to-cap clearance and any pocket walls that affect finger approach. | No current value | |
| [ ] | Cap resting orientation | Record whether each cap is plug-down, flange-down, or otherwise constrained. | Simulation assumes a usable loose cap; physical orientation unverified | |
| [ ] | Cap extraction clearance | Test the intended finger approach without grasping, then record nearest fixture interference. | No current value | |
| [ ] | Pickup confirmation observation | Define the visible TableCam or wrist-camera evidence that a cap left its pocket. | No current rule | |
| [ ] | Inventory confirmation observation | Define how the system counts available caps after each attempted pickup. | No current rule | |

## 6. Required capture set for the first physical calibration session

1. One native TableCam colour image and matching raw depth image with the empty tray fixture.
2. One native TableCam colour image and matching raw depth image with all vials and caps seated.
3. One image/depth pair for each of: a missing cap, a stable crooked cap, and a usable replacement cap.
4. Six TableCam calibration-reference images spanning the tray and supply area, with measured robot-base coordinates entered in the capture template.
5. At least six physical tool-center-point records for each of pickup, placement, and pressing.
6. Close-up photos and measured dimensions of both installed custom fingers and their moving jaw carriers.
7. Ten tray remove/replace captures to measure repeatability.
8. A photo and full measurement record for the replacement-cap supply fixture.

## Acceptance before physical manipulation

Do not enable automated pick, place, or press actions until the following are
recorded: measured table-camera accuracy over the full work area; measured tool
center points; verified gripper gap and groove geometry; measured tray pose
repeatability; a defined replacement-cap fixture; and supervised single-action
tests with reinspection. The hackathon mock check verifies the software action
format only; it does not replace these measurements.
