#pragma once
// ---------------------------------------------------------------------------
// nuka::scene — authored scene METADATA that is NOT a cook-fidelity record:
// the per-articulation settled IC (initial_state) and the union GRASP config
// block (the externalized h1_union_scene_factory constants). Stored on the
// SceneIR alongside the records; persisted by the .nks Save/Load (M7 T3);
// consumed by the union-slot cook CookSceneToUnionTemplate (M7 T4).
//
// WHY A SEPARATE BLOCK (not 34 tree nodes). The union-specific CONTACT GEOMETRY
// (30 wrap spheres + 4 foot spheres bound to imported H1 links by NAME+OFFSET)
// and the DRIVE TABLES are authoring CONFIG, not scene-graph objects: they bind
// to links the H1 import owns, by name, and reproduce the factory's
// BatchedSceneTemplate. The .nks `tree` carries the first-class objects (the H1
// import, the cup body, the table body); this block carries the factory's ~120
// empirical constants so T4 can rebuild the union template WITHOUT the factory.
//
// HOST-ONLY, NO CUDA, NO nk/phi deps (plain data; std + math only).
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace nuka::scene {

// The settled IC of ONE articulation (the factory's 200-step settle product,
// controller ruling R4: BAKED — never re-derived). `qpos` is the flat PER-LINK
// q array (length == the articulation's device link count; the floating base
// root link's slot carries the base DOFs implicitly, the rest are 1-DOF joints)
// and `root` is the settled base pose. T4 maps qpos -> model.articulation.
// initial_q and root -> the base pose. Keyed in SceneInitialState by the
// articulation's tree node path (e.g. the import attach prefix "h1").
struct ArticulationInitialState {
    std::vector<float> qpos;                          // per-link settled q.
    math::Transform    root = math::Transform::Identity();
};

// initial_state section: node-path -> settled IC. A std::map keeps the keys in a
// deterministic (lexicographic) order so a second Save is byte-identical.
using SceneInitialState = std::map<std::string, ArticulationInitialState>;

// One wrap / palm / foot contact-sphere descriptor BOUND TO AN IMPORTED H1 LINK
// BY NAME (resolved to a device link at cook). `local_offset` is the sphere
// center in the link frame; `region` groups finger/thumb/palm/foot (diagnostic).
// The factory authors these as WrapSpheres(...) / the foot toe-heel loop; T4
// re-resolves link_name -> link index against the cooked H1 and emits one
// FingerSphereHull (wrap/palm) or FootSpherePlane (foot) union slot per sphere.
struct GraspContactSphere {
    std::string link_name;            // imported H1 link the sphere rides on.
    math::Vec3  local_offset{};       // sphere center in the link frame (m).
    float       radius = 0.0f;        // sphere radius (m).
    std::string region;               // "finger" | "thumb" | "palm" | "foot".
    uint32_t    broadphase_id = 0u;   // the factory's disjoint handle (9000+/12000+).
};

// One reference PD drive-table entry, keyed by the imported H1 link NAME (T4
// re-resolves to a device link + its flat DOF column). Mirrors
// H1UnionDriveEntry but stores the link by NAME (cook-portable) instead of a
// device index. tau = kp*(target - q[link]) - kd*qdot[link], clamped to +/-tlim
// when tlim > 0; `grip` marks the 12 wrap-driven close links (BITE kill-switch).
struct GraspDriveEntry {
    std::string link_name;
    float       target = 0.0f;
    float       kp     = 0.0f;
    float       kd     = 0.0f;
    float       tlim   = 0.0f;        // physical |tau| clamp (0 -> unclamped).
    uint8_t     grip   = 0u;          // 1 == a wrap-driven grip-close link.
};

// The union GRASP config block — the factory's externalized constants. Every
// field is a faithful copy of a h1_union_scene_factory.{hpp,cpp} constant or a
// resolved table; T4's CookSceneToUnionTemplate reads ONLY this block + the
// tree (H1 import + cup body + table body) to rebuild the union template.
struct GraspConfig {
    bool present = false;   // false -> no grasp block authored (default scene).

    // -- integrator constants the settle ran at (validated; R4) --------------
    float gravity_z = -9.81f;          // kH1UnionGravityZ.
    float dt        = 1.0f / 240.0f;   // kH1UnionDt.

    // -- friction coefficients ----------------------------------------------
    float finger_mu = 0.8f;            // kMu — finger x cup.
    float table_mu  = 0.6f;            // table_mu — cup-proxy x table.
    float foot_mu   = 0.8f;            // foot_mu — feet x ground.

    // -- the cup -------------------------------------------------------------
    float cup_mass   = 0.2f;           // kCupMass.
    float cup_scale_xy = 1.8f;         // kSxy.
    float cup_scale_z  = 1.8f;         // kSz.
    // The EXACT scaled (1.8x) + COM-centered cup hull verts the factory built
    // (ScaleCupHull -> COM-recenter): flat x,y,z triples, MESH-LOCAL. This is
    // CoResidentCup.hull_verts 1:1. Offloaded to the sibling .nka as a SAMP
    // chunk (a vertex cloud, no topology); `cup_hull_ref` is the AssetRef text.
    std::vector<float> cup_hull_verts;
    std::string        cup_hull_ref;   // ".nka#SAMP/<i>" (set by Save/Load).
    uint32_t           cup_broadphase_id = 7000u;  // CoResidentCup.broadphase_body_id.

    // -- the contact spheres (wrap + palm + foot), bound to H1 links by name --
    // wrap/palm: FingerSphereHull slots (cup); foot: FootSpherePlane slots
    // (ground). The factory authors 30 wrap (no palm in the GO set) + 4 foot.
    std::vector<GraspContactSphere> wrap_spheres;   // region finger/thumb/palm.
    std::vector<GraspContactSphere> foot_spheres;   // region foot.

    // -- ground + table statics ---------------------------------------------
    float    ground_height       = 0.0f;     // CoResidentGround.height.
    uint32_t ground_broadphase_id = 8000u;   // CoResidentGround.broadphase_id.
    float    table_height        = 0.0f;     // settled cup_bottom + 2mm.
    uint32_t table_broadphase_id = 8500u;
    // The cup-table proxy box (the flat-bottom proxy; bbox half-extents of the
    // scaled hull, centered): BodyBoxPlane slot vs the table.
    math::Vec3 cup_table_proxy_half{};
    math::Vec3 cup_table_proxy_offset{};
    uint32_t   cup_table_proxy_id = 7001u;

    // -- the 12 grip links (the close drive + BITE kill columns), by name ----
    std::vector<std::string> grip_links;     // kWrapDriven[12] (resolved order).

    // -- the three reference drive tables (hold/rest/close), by link name ----
    // hold = the settled curl held; rest = wrap backed off kH1UnionRestBackOffset;
    // close = the H1.1 close PD at +kH1UnionCloseOffset. Numerous; stored as
    // arrays. T4 re-resolves link_name -> device link + DOF column.
    std::vector<GraspDriveEntry> drive_hold;
    std::vector<GraspDriveEntry> drive_rest;
    std::vector<GraspDriveEntry> drive_close;

    // -- per-DOF |torque| limits (length == dof_stride; 0 on the 6 base cols) --
    // The "random torques within per-joint limits" envelope. Indexed by the flat
    // action/obs DOF column (prefix-sum DofIndexOf order).
    std::vector<float> dof_torque_limit;

    // -- scene scalars (force-balance + shape metadata) ----------------------
    double   total_mass = 0.0;         // Σ link masses (m*g*dt reference).
    uint32_t dof_stride = 0u;          // 51 == 6 base + 45 joints.
    uint32_t base_dof   = 0u;          // 6 (FloatingBase root).
    float    poly_cx    = 0.0f;        // seat-time foot-polygon center x (CoP ref).
    float    ankle_settle_tau[2] = {0.0f, 0.0f};  // the settle's final ankle CoP tau.

    // The proven grip choreography offsets (rad) (surfaced for T4 / python).
    float close_offset     = 0.18f;    // kH1UnionCloseOffset.
    float rest_back_offset = -0.25f;   // kH1UnionRestBackOffset.
};

}  // namespace nuka::scene
