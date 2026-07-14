# BDX 一镜到底场景施工规格(controller 设计,2026-07-02)

参照 `bdx-sb.md` 四分镜,owner 要求:**一镜到底**(单连续布景+单连续运镜)、背景环境美化。
所有物理走 ONE 通用路径;demo 构建 python 免编译优先;产物进 /data/xtzhang25/_work/activate/。

## 0. 主角与尺度
- BDX = examples/scenes/bdx_stand.nks(open_duck_mini_v2,站高 ~0.35m,base ~0.20m,质量 2.1kg,capsule 脚)。
- 需补碰撞授权:head_assembly 1×sphere/capsule(D01 顶布、D04 撞摆锤),trunk 1×box(踉跄防穿)。每 link 限 1 geom。
- 场景尺度全部对 duck 缩放(GTC 原片是人尺度 BDX,我们按 0.35m duck 等比 ~0.5×)。

## 1. 连续布景(世界坐标,+z up,米;行进方向 +x)
按行进顺序四区,总长 ~6m,宽 ~1.2m 通道,两侧墙/道具作 set dressing:

**Zone A 木梯+门框垂布(D01)** x∈[0, 1.2]
- 木楼梯下行:5 阶,rise 0.04,run 0.14,宽 0.7(顶部平台 x<0,z=+0.20 → 落地 z=0)。
- 阶底 x≈0.9 处木门框:两柱 + 横梁(梁底 z=0.46);布 0.5×0.35m XPBD cloth 顶边 pin 在梁上,自然垂到 z≈0.26(duck 头 0.35 → 必顶起布)。
- 物理:楼梯=static box 群;布=MediaRecord Cloth×Xpbd(已有);布↔头/身双向耦合(已有 coupling rows)。
- 踉跄 beat:最后一阶 rise 加大到 0.055(诚实动力学扰动,靠 S2 抗扰策略恢复,不做假动画)。

**Zone B 碎石地 MLS-MPM(D02)** x∈[1.4, 3.0]
- 浅槽 1.6×0.9m、深 0.035m,内填 granular MPM 床(粒径 ~5mm,预估 15-40 万粒,按显存/帧率实测收敛)。
- 物理:**新 model_kind=4 Drucker-Prager granular**(见 §3 缺口 G1)+ 既有 MPM↔rigid 耦合;验收=脚印持久凹痕 + 石子被挤开/溅散不回弹成液体。

**Zone C 小物件散落区(D03)** x∈[3.2, 4.2]
- 硬质地面上散落 ~40 个自由刚体:鹅卵石(凸包 8-15mm)、螺丝/螺栓(单凸包近似,~3-8g)、垫圈。摆位密集在行进线上,脚摆动自然勾飞。
- 物理:纯刚体接触(已有);验收=被踢物翻滚/弹跳/互不穿模、静置物不抖。

**Zone D 绳吊重物(D04)** x∈[4.4, 5.6]
- 头顶横梁(z=0.75)垂绳吊木块(0.12×0.08×0.05m,~0.3kg),块底 z≈0.30(撞 duck 头)。
- 绳=8-12 节细 capsule 链 + ball joint(通用关节行,无专用绳求解器);顶端锚横梁,底端接块。
- 验收=撞击后沿弧摆动、动量传递让 duck 头后仰+找平(S2 抗扰),绳不拉断不抖。

**Set dressing(owner 要求"好看的环境")**
- 仓库/工坊基调:两侧木板墙+立柱、背景木箱/木桶/工具架、Zone A 顶部暖主光(门框布半透光)、冷色补光;地面材质分区(木板→碎石→水泥→水泥)。
- 全部 static 刚体+visual mesh/图元,合理 poly 预算;灯光走 studio_beauty/离线 RT 已有能力,媒介色 gap 见 §3 G5。

## 2. 一镜到底运镜
单条相机样条 + look-at 目标 + **时间重映射**(不切镜):
- beat1(Zone A):低机位侧面近景,轻微跟随下移;
- beat2(Zone B):降到贴地,侧后 45° 跟拍下半身,平移;
- beat3(Zone C):推近到微距,**时间减速 ~0.25×**(sim 照常,渲染帧率×4 重采样),浅景深(§4 光追演进项,draft 阶段可无);
- beat4(Zone D):拉回侧面中景,静止微摇。
实现:python 相机路径描述(关键帧+Catmull-Rom+time-remap 曲线),喂离线渲染器逐帧渲。

## 3. 引擎/易用性缺口(全部通用修,禁特化)
- **G1 granular 本构(最长杆,先行)**:mpm.cu 本构加 model_kind=4 Drucker-Prager(弹性预测+DP 屈服面塑性回映到 F;dp_friction/dp_cohesion 材质列已 plumbed 未消费);MediaRecord Kind 加 Granular(×MlsMpm);nks/门面/validator 跟上。gate:方料坍塌成堆(休止角≈摩擦角)、脚印持久、Go2 金丝雀+cloth FNV+MPM 既有字节门不动。
- **G2 头/躯干碰撞授权**:bdx_author.py 加 head sphere + trunk box(数据授权,非代码)。
- **G3 绳链 authoring**:python Scene API 补通用 verbs(add_collision_shape/add_joint/initial_position/is_static/terrain——正好补上 #2 阶段记录的四缺口);绳=数据(capsule 链+ball joints)。
- **G4 相机路径+时间重映射**:渲染工具层(python),不动引擎。
- **G5 媒介渲染色**:studio_beauty 目前媒介面固定 kMatCloth 色,render_material_id 不生效(usability memory 已记)→ 布/碎石要能上色。
- **G6 MPM 域与 rigid 世界对齐**:MPM 槽域边界(separating wall)与 Zone B 槽几何一致,+z 开放。

## 4. 里程碑与验收(每个渲染里程碑控制器亲自 Read 关键帧逐缺陷)
- W1=G1 granular 落地+字节门+ultracode 对抗审查(引擎里程碑,owner 规矩)。
- W2=全布景 .nks(python authoring)+PD 慢走占位穿场;分区静帧控制器过目。
- W3=行走策略接管全程 choreography(命令沿路径),四 beat 物理探针(布顶起高度/脚印深度/被踢物速度/摆锤幅度)。
- W4=一镜到底渲染:draft 540p 全片 → 修缺陷 → 1080p 成片;光追演进(#6:景深/慢动作插帧/降噪)接在 draft 后。
