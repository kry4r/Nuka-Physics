#!/usr/bin/env python3
"""Check which contact family is configured."""
import nuka
from pathlib import Path
from nuka.tasks.libero_black_bowl import LiberoBlackBowlController

REPO = Path(__file__).parent
SCENE = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.nks"

with nuka.Device.create(0) as device:
    controller = LiberoBlackBowlController(
        SCENE, device, dt=0.005,
        control_backend="joint_pd",
        render_quality="preview",
    )
    world = controller.world
    
    # Try to access contact family setting
    print(f"World created, env_count={world.env_count}, dt={world.dt}")
    
    # The contact family should be visible in world properties
    if hasattr(world, 'contact_family'):
        print(f"contact_family attribute: {world.contact_family}")
    
    # Check scene metadata
    print(f"\nWorld attributes: {[a for a in dir(world) if not a.startswith('_')]}")
