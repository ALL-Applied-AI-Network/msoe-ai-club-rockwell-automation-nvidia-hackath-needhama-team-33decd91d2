#!/usr/bin/env python3
"""Compatibility entry point for the canonical, physics-qualified scene builder."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from src.remediation.qualify_scene import parser,run

if __name__=='__main__':
    p=parser()
    p.add_argument('--good',type=int)
    args=p.parse_args()
    if args.good is not None and args.good+args.crooked+args.missing!=30:
        p.error('good + crooked + missing must equal 30')
    if args.steps<120: p.error('At least 120 physics steps required')
    if not run(args)['accepted']: raise SystemExit(2)
