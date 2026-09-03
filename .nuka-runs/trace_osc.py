import json, sys, torch, nuka
sys.path.insert(0, 'python')
import nuka.tasks.libero_black_bowl as lbb

SCENE = '.nuka-assets/generated/libero/libero_spatial_black_bowl.nks'
with nuka.Device.create(0) as device:
    c = lbb.LiberoBlackBowlController(SCENE, device, control_backend="osc")
    c.reset()
    for _ in range(100):
        c.step()
    eef0 = c.eef_pose()[:3].tolist()
    bowl = c.target_position().tolist()
    print('after settle eef', [round(v,4) for v in eef0])
    print('bowl           ', [round(v,4) for v in bowl])
    # Command a pure +X move at full scale for 40 control steps, no policy.
    act = torch.zeros(7, device=c.q.device); act[0] = 1.0; act[6] = -1.0
    for k in range(40):
        c.set_policy_action(act)
        for _ in range(10):
            c.step()
        if k % 10 == 9:
            e = c.eef_pose()[:3].tolist()
            t = c.task_target[0].tolist()
            print('k%-3d target %s  eef %s' % (k+1,
                  [round(v,4) for v in t], [round(v,4) for v in e]))
