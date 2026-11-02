"""Convert the supplied custom gripper STEP model to USD using Isaac's CAD converter.

Conversion preserves the authored CAD geometry only.  It does not choose a wrist
mounting pose, create jaw articulation, or approve collision/contact use.
"""
import argparse
import asyncio
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=Path, default=ROOT / 'data/gripper.step')
    parser.add_argument('--out', type=Path, default=ROOT / 'data/gripper.usd')
    parser.add_argument('--report', type=Path, default=ROOT / 'runs/archive/2026-10-21_remediation/gripper_cad_conversion.json')
    args = parser.parse_args()
    if not args.step.is_file():
        raise FileNotFoundError(args.step)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True})
    try:
        import omni.kit.app
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate('omni.kit.converter.hoops_core', True)
        for _ in range(20):
            app.update()
        from omni.kit.converter.hoops_core import get_instance, is_format_supported
        if not is_format_supported(str(args.step)):
            raise RuntimeError(f'Isaac CAD converter does not support {args.step.suffix}')
        converter = get_instance()
        if converter is None:
            raise RuntimeError('Isaac HOOPS CAD converter did not initialize')

        async def convert():
            return await converter.create_converter_task(str(args.step), str(args.out), {})

        _, status = asyncio.run(convert())
        report = {
            'input': str(args.step.relative_to(ROOT)),
            'output': str(args.out.relative_to(ROOT)),
            'converter': 'omni.kit.converter.hoops_core',
            'error_code': int(status.error_code),
            'error_message': str(status.error_msg),
            'output_exists': args.out.is_file(),
            'limitations': 'USD conversion has not established wrist mounting pose, jaw motion, collision proxies, TCP, or contact approval.',
        }
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        if status.error_code != 0 or not args.out.is_file():
            raise RuntimeError(f'CAD conversion failed: {status.error_msg}')
        print(json.dumps(report))
    finally:
        app.close()


if __name__ == '__main__':
    main()
