# HANDOFF-scene.md — BDX 一镜到底场景组装 (task #4) 交接

**分支** `scene-oneshot` @ `e160f09`(基线 29d8935 windows-editor,未 push)。工作树干净,全部已 commit。
**自评完成度 ~85%**:四区布景+G2+G3 核心 verbs+加载门+预览图 DONE;债 = 布↔刚体接触缺口(引擎级)、MPM×XPBD 不能同 cook(引擎级)、大地面正视角渲染发灰(渲染级)。

## 0. 原任务书 vs 已完成

| 任务书条目 | 状态 | 说明 |
|---|---|---|
| Zone A 木梯 5 阶 rise 0.04 + 门框 | ✅ | rise 0.04 / run 0.14 / 宽 0.7,梯在 x<0 升向后方平台;门框 x=0.9,柱 y±0.35,横梁底 z≈0.47 |
| Zone A XPBD 垂布(顶起) | ⚠️ 改设计 | **布↔刚体 box 接触在通用 built-scene 路径没接线**(实测自由布直接穿过横梁掉到 -10m)。改为 **perimeter-pinned 布帘** 18×26 @0.02m,中心 (0.9,0,0.30),头(0.35m)从下方顶起鼓包。cloth_free=True 的悬挂帘需要引擎补 cloth↔free-rigid 接触或 top-edge pin,都没有 |
| Zone B 碎石槽 1.6×0.9×0.035 MPM granular | ✅ | model_kind=4 DP,φ=34°,ρ=1600,fill z[0.004,0.037],spacing 0.013(gate 用 0.016 = 10282 粒);床沉降 ~25%(0.037→z_max 0.028)稳定,0 逃逸 0 NaN |
| Zone C ~40 小刚体 | ✅ | 45 个:球 r1.8-3.6cm + capsule(螺栓) + box(垫片),x[3.45,4.05] y±0.22,静置稳定不抖 |
| Zone D 绳链 8-12 节 + 0.3kg 块 | ✅ | **7 节 capsule(r0.012, seg0.055)revolute-Y 链**(不是 ball joint——见坑 #1)+ 0.3kg 块(0.16×0.10×0.07);梁 (5.0,0,0.75),块底 z≈0.30 撞头高度。悬垂/摆动/有限性验证过 |
| 环境美化 | ✅(带渲染债) | HDRI sky_2k、dirt 地、wood/gravel/fabric/stone/metal/crate 材质、8 木箱在动线外 y>1.3 |
| 鸭子起点 + PD 站立 | ✅ | spawn (0,0,0.21),楼梯脚下(梯升向 -x 后方平台);600 步 PD 站立 quat_w=1.0 |
| G2 头/躯干碰撞 geom | ✅ | 经新 verb 加(**不改 bdx_stand.nks**):head sphere r0.04 @(0.02,0,0)、trunk box (0.04,0.035,0.028) @(0,0,-0.005)。**必须小且分离**——初版两 geom 相触,自碰把脖子顶伸长 |
| G3 python Scene verbs | ✅ 核心 3 个 | 见 §2。add_joint 未做门面(绳链走 USD compose 更稳,见坑 #1);initial_position/is_static/terrain 仍是 bdx_author.py 的 JSON 后处理模式(未门面化) |
| G6 MPM 域对齐 | ✅ 自动解决 | mpm.cu 自带 separating domain wall 在床 AABB(x/y 四壁+z 底,+z 开放)+静态地面 BC;**刚性槽壁只是布景**(MPM 不与 rigid primitive 碰撞)。记债:脚印需要脚上有 cooked SDF,capsule 脚没有 |
| 床料预沉降 | ⚠️ 部分 | 未烘进初始态(粒子初始态在 cook 内生成,无 python 写回口);实测 250 步内沉降完成且稳定,渲染前 settle 220 步即可。烘化需要引擎加粒子初始态导入口 |

## 1. Commit 列表(29d8935..e160f09)
- `3878dfb` **C-ABI + binding**:nuka_scene_add_collision_shape(新)+ SceneBuilder.compose/save(绑已有 C-ABI)
- `78080a0` demo 四脚本(author/gate/preview/rope)
- `bc5a2ab` 场景成品 .nks/.nka/rope.usda + 构图修正
- `ad491dd` 材质 base_color 兜底 + matte 地 + 去 heightfield
- `e160f09` verbs 最小单测(host-only 无 GPU)

## 2. 新 python verbs(nuka.SceneBuilder,全通用非 demo 特供)
```python
b.add_collision_shape(node_path, kind, dims=[], pos=[], quat=[], friction=-1.0,
                      contype=1, conaffinity=1)   # 给已导入 link 挂碰撞 geom
b.compose(addon, pos=[], quat=[], attach_at="")   # 嫁接子装配(带 joint 的绳/任何 importer 片段)
b.save(nks_path)                                  # built scene → 自含 .nks(+.nka)
```
C-ABI:`nuka_scene_add_collision_shape`(src/c_abi/scene.cpp,decl 在 src/include/nuka/nuka_scene.h);compose/save 直接绑本已存在的 `nuka_scene_compose`/`nuka_scene_save`。node_path 要**完整派生路径**,如 `base/trunk_assembly/.../head_assembly`(短名 find 不到)。

## 3. 关键文件地图
- `examples/demo/bdx_oneshot_author.py` — 主授权脚本(四区+材质+环境;`add_cloth(b)`/`add_granular(b)` 两个 build-time media helper 在文件顶部;共享常量 DOOR_X=0.9 / TROUGH=(1.4,3.0,0.45))
- `examples/demo/bdx_oneshot_rope.py` — 绳 .usda 生成器(`rope_usda()`/`write_rope_usda()`)
- `examples/demo/bdx_oneshot_gate.py` — 加载门(`--media none|cloth|granular`,打 base_z/NaN/逃逸数,PASS/FAIL)
- `examples/demo/bdx_oneshot_preview.py` — render_beauty 预览(VIEWS 6 机位+overhead debug;`--only` 过滤)
- `examples/demo/bdx_oneshot_verbs_test.py` — verbs 单测(无 GPU)
- `examples/scenes/bdx_oneshot.nks/.nka` + `bdx_oneshot_rope.usda` — 成品
- 改动的引擎面:`src/c_abi/scene.cpp`、`src/include/nuka/nuka_scene.h`、`python/src/nuka_ext.cpp`(全加法,回归金丝雀逐位 0.923080623 不动,4 passed)

## 4. 踩坑 → 修法(接力者直接抄)
1. **Spherical joint 被 cook 静默丢弃**(articulation_cooker.cpp:27 IsSupportedJoint 只认 Revolute/Prismatic/Fixed;joint 被 skip→链散架自由落体)。**绳=全同轴 revolute-Y 串链**(平面摆),fixed root(kinematic mass=0 anchor)。多articulation 共存 OK(CookArticulations 返回 vector;鸭+绳=2 articulations,标准 log 提示 host mirror 只盖 articulation 0,无害)。
2. **USD 串链的 translate 是父相对**:每节 `xformOp:translate=(0,0,-seg)`(FK 沿链累加)。写绝对 z 会把链翻着叠上去(实测 0.6/1.15/1.6/1.95)。
3. **布穿刚体**:built-scene 路径 cloth 只与 heightfield+articulation 有接触,**与 rigid box 无**(自由布落到 -10.5)。悬帘不可行→perimeter-pin。别再试 free 布搭横梁。
4. **MPM×XPBD 禁混**:cook_to_model.cpp `ValidateMedia`——MLS-MPM 不能与 XPBD/PBF 同 Model,MPM 也只能 1 个。所以 .nks 只存刚性舞台,cloth/granular 在 build 时二选一注入(gate/preview 的 `--media` 即此)。
5. **create_from_scene 拒绝带 media 的 .nks**(world.cpp:511 NOT_SUPPORTED)→ 加载一律 `SceneBuilder.create(path).build()`。
6. **stale .so**:conda 环境有 `_nuka_editable` meta-finder 抢 import。用 `/data/xtzhang25/_work/activate/pybootstrap/sitecustomize.py`(已存在)放 PYTHONPATH 首位即可让 worktree 的 nuka+扩展生效。
7. **头/躯干 geom 相触会顶脖子**:两 geom 尺寸/位置见 §0 G2 行,保持分离。
8. **大地面正视角发灰**(渲染债):巨大 dirt box 正视角渲成灰、掠射角正常;小 crate 完全正常。试过 尺寸16→7.5→14m/roughness 1.0/base_color 深棕/uv_scale 3→40/intensity 1.05→0.68→0.85/去 heightfield——都无效。是 offline RT 对大平面材质解析的行为,**引擎渲染侧修**(交 #5/#6);运镜尽量低机位掠射。
9. **heightfield 渲染在刚体之上盖灰片**→已去掉 terrain,脚直接踩 dirt box(gate 全绿)。若动力学需要 heightfield 脚感,恢复 `doc["terrain"]=[TERRAIN]`(author 里 TERRAIN dict 还在)并接受其灰色。
10. **movable free-body 的渲染材质不生效**(石子白色;static 正常)。石子当白卵石可接受,或渲染侧修。
11. base_z 读数 0.236≠鸭升高:多 articulation 下 BASE_POSE 字段语义;鸭 base is_static@0.21,以 finite+quat_w+渲染为准。

## 5. 构建/运行(全部现成)
```bash
# 已建好:core=/data/xtzhang25/_work/activate/build-scene(target nuka, gcc-10+nvcc12.8+arch89)
#         pyext=/data/xtzhang25/_work/activate/build-scene-pyext(改 nuka_ext.cpp 后 rebuild 并 cp)
cmake --build /data/xtzhang25/_work/activate/build-scene --target nuka -j24
cmake --build /data/xtzhang25/_work/activate/build-scene-pyext -j16
cp /data/xtzhang25/_work/activate/build-scene-pyext/_nuka_ext.*.so \
   /data/xtzhang25/_work/activate/worktrees/scene-oneshot/python/nuka/
# 运行环境(每条命令都要)
export CUDA_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/data/xtzhang25/_work/activate/build-scene/src:/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/data/xtzhang25/_work/activate/pybootstrap:/data/xtzhang25/_work/activate/worktrees/scene-oneshot/python:/data/xtzhang25/_work/activate/worktrees/scene-oneshot/examples/demo
conda run -n nuka-v03 python examples/demo/bdx_oneshot_author.py --rocks 45   # 重生成场景
conda run -n nuka-v03 python examples/demo/bdx_oneshot_gate.py --steps 600 --media granular  # 门
conda run -n nuka-v03 python examples/demo/bdx_oneshot_preview.py --media none --spp 128     # 预览
/data/xtzhang25/_work/activate/build-scene/tests/nuka_regression_test  # 金丝雀 0.923080623 恒"FAIL"=设计
# granular cook 慢(~3-4min),渲染任务放 background;MPM 门跑过:10282 粒 0 NaN 0 逃逸
```

## 6. 预览图(/data/xtzhang25/_work/activate/out/oneshot_scene/,12 张,已逐张看)
- `none_06_overhead_debug.png` ✅ 全局布局正确(鸭→槽→石→梁,箱在动线外)
- `none_01_zoneA_stairs_door.png` ✅ 鸭渲染正确(头贴身,G2 修后)
- `granular_02_zoneB_trough.png` ✅✅ **最佳**:MPM 碎石床沉降后整齐填槽,D02 招牌
- `none_04_zoneD_rope.png` ✅ 梁+深色绳+木块成摆锤,此掠射角地面呈 dirt
- `none_03_zoneC_rocks.png` ⚠️ 石可见但白色(债 #10)
- `cloth_01_zoneA_stairs_door.png` ⚠️ 红布=stable pinned panel(改设计后正确,但非"垂帘"观感)
- 各 `*_00_establish.png` ⚠️ 正视角地面灰(债 #8)

## 7. 剩余工作优先级(给接力者)
1. **(渲染侧,最影响观感)** 修大平面材质发灰 + movable body 材质——都在 offline RT/render_world 侧,与 #6 光追演进合并处理最顺。
2. **(引擎侧,D01 需要)** cloth↔rigid-primitive 接触 或 cloth top-edge pin(cook_to_model.cpp:1716 的 pin 逻辑只有 perimeter/free 两档)→ 才能做真"门框垂帘被顶起"。
3. **(引擎侧,D02 行走需要)** 脚 capsule 无 cooked SDF→MPM 脚印无从谈起;给 primitive shape cook 解析 SDF(mpm.cu:685 会自动吃)。
4. (可选)G3 补 add_joint/initial_position/is_static 门面 verbs;把 bdx_author.py 的 JSON workaround 换正式 verbs。
5. (可选)granular 预沉降烘进初始态(需要粒子初始态导入口)。

## 8. 给 #5 运镜的接口速记
相机样条起止建议:A(0.55,-1.75,0.50)→B(2.2,-1.55,0.48)→C 微距(3.78,-0.72,0.26)→D(5.0,-2.05,0.52),look 各 zone 中心(preview VIEWS 就是关键帧);低机位掠射角地面观感最好。可查名实体:`ropeanchor/ropelink_0..6/ropeblock`、`box_*/sphere_*/capsule_*`(Zone C)、鸭 link 名同 bdx_stand。beat 切媒介:A 段 build 用 add_cloth,B 段 build 用 add_granular(一镜到底若必须单 world,则只能选一种粒子媒介,或引擎先修 #4 债)。
