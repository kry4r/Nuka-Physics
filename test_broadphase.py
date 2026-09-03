#!/usr/bin/env python3
"""Test to check broadphase pair detection."""
import sys
sys.path.insert(0, './build-win-editor/python')

import nuka
import numpy as np

with nuka.Device.create(0) as dev:
    # Load the pi0.5 demo scene
    w = nuka.World.load(dev, 'demos/pi05_libero_demo/pi05.nukaworld.yaml')

    # Step once to populate broadphase
    w.step()

    # Get contact info
    info = w.contact_info()
    slot_count = info['slot_count']
    slot_stride = info['slot_stride']
    env_count = w.env_count

    print(f"=== Broadphase Analysis ===")
    print(f"slot_count: {slot_count}")
    print(f"slot_stride: {slot_stride}")
    print(f"env_count: {env_count}")
    print(f"rigid_slot_cap: {info.get('rigid_slot_cap', 'N/A')}")

    # Try to read broadphase pair count
    # This might not be exposed, but let's check what fields are available
    print(f"\nAvailable contact_info fields:")
    for key in sorted(info.keys()):
        val = info[key]
        if isinstance(val, (int, float)):
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: {type(val).__name__}")
