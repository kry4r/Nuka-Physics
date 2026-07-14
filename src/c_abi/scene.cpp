// ---------------------------------------------------------------------------
// nuka::c_abi -- C ABI for the GENERIC SCENE-AUTHORING surface (M9 T4).
//
// A SceneRecord owns ONE in-memory nuka::scene::SceneIR (the M2b facade). This
// TU loads any supported format (mjcf/urdf/usd/nks) through the SAME
// LoadSceneByExtension dispatch world.cpp uses, composes / edits / settles /
// saves the scene, and never special-cases grasp / union / any demo (owner
// [[unified-world-no-special-grasp-binding]]): behavior comes from the imported
// scene DATA + a per-scene control SCRIPT, not from this API.
//
// FACADE HAZARD (M2b). The SceneIR facade re-projects tree_ + ecs_ (fresh
// entities + SceneNode pointers) AFTER any record mutation (a Get*Mut marks it
// dirty -- scene_ir.hpp). So we NEVER cache a node/entity across a mutating
// call: each edit RE-RESOLVES the node by string path, reads the current facade
// to map path -> record id, then mutates the record; the next facade read
// lazily re-projects. This is also why nuka_scene.h addresses nodes by PATH
// (stable) instead of an opaque node handle (invalidated by the next edit).
//
// Mirrors the recorder.cpp / union_world.cpp per-module pattern: a file-local
// HandleTable, null-guard at every entry, try/catch -> MapExceptionToResult,
// zeroed out-params on failure. Only nuka_scene_settle touches a device (the
// device's OWNED phi v2 backend, mirroring recorder.cpp's cook-to-nk::World);
// everything else is host-only with ZERO CUDA tokens.
// ---------------------------------------------------------------------------

#include "nuka/nuka_scene.h"

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/cook/settle.hpp"
#include "scene/ecs/components.hpp"
#include "scene/ecs/registry.hpp"
#include "scene/format/nks.hpp"
#include "scene/graph/scene_graph.hpp"
#include "scene/scene_compose.hpp"
#include "scene/scene_ir.hpp"
#include "scene/scene_map.hpp"
#include "scene/scene_metadata.hpp"

#include <cstdint>
#include <exception>
#include <filesystem>
#include <memory>
#include <new>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::c_abi {

// SceneRecord ctor/dtor/move are out-of-line here (where SceneIR is complete);
// the unique_ptr<SceneIR> member needs the full type for ~unique_ptr.
SceneRecord::SceneRecord() = default;
SceneRecord::~SceneRecord() = default;
SceneRecord::SceneRecord(SceneRecord&&) noexcept = default;
SceneRecord& SceneRecord::operator=(SceneRecord&&) noexcept = default;

// The built-scene handle table (declared in handle_table.hpp). Shared so the
// scene-builder TU + the built-scene world create reach the SAME SceneRecord.
HandleTable<nuka_scene_t, SceneRecord>& SceneTable() {
    static HandleTable<nuka_scene_t, SceneRecord> table;
    return table;
}

}  // namespace nuka::c_abi

namespace {

using nuka::c_abi::DeviceRecord;
using nuka::c_abi::SceneRecord;
using nuka::c_abi::SceneTable;  // the shared table (defined in nuka::c_abi above).
namespace cook = nuka::scene::cook;
namespace nks = nuka::scene::nks;
namespace nphi = nuka::phi;
namespace nk = nuka::nk;

// Dispatch by file extension to the right importer / .nks loader -- the SAME
// dispatch world.cpp's create path uses (kept in sync deliberately; this is the
// ONE generic entry). Throws on a missing file (the importer / nks::Load do) or
// returns false on an unrecognized extension.
bool LoadByExtension(const std::string& path, nuka::scene::SceneIR* out) {
    const std::filesystem::path p(path);
    const std::string ext = p.extension().string();
    if (ext == ".nks") {
        *out = nks::Load(path);
        return true;
    }
    if (ext == ".xml" || ext == ".mjcf") {
        *out = nuka::import::LoadMjcf(path);
        return true;
    }
    if (ext == ".urdf") {
        *out = nuka::import::LoadUrdf(path);
        return true;
    }
    if (ext == ".usd" || ext == ".usda") {
        *out = nuka::import::LoadUsd(path);
        return true;
    }
    return false;  // unknown extension -> INVALID_ARG at the caller.
}

// Build a Transform from optional pos[3] (xyz) + quat[4] (w,x,y,z). A NULL
// component leaves that part Identity (the caller documents "unchanged" on a
// per-entry basis when it merges into an existing transform).
nuka::math::Transform MakeTransform(const float pos[3], const float quat[4]) {
    nuka::math::Transform t = nuka::math::Transform::Identity();
    if (pos != nullptr) {
        t.position = nuka::math::Vec3{pos[0], pos[1], pos[2]};
    }
    if (quat != nullptr) {
        t.rotation = nuka::math::Quat{quat[0], quat[1], quat[2], quat[3]};
    }
    return t;
}

// Resolve a derived tree path to its RECORD ids by reading the CURRENT facade
// (call AFTER any pending mutation so the facade is fresh). A node maps to at
// most one body row and/or one shape row via the SceneMap-free record<->entity
// inverse (EntityOfBody / EntityOfShape close the loop to a node + path). We
// scan bodies then shapes because a node is exactly one of {group, body, shape}
// and PathOf is the authority. Returns {body_id, shape_id} with kInvalid* when
// the path does not resolve to that record kind.
struct PathRecords {
    nuka::scene::BodyId  body  = nuka::scene::kInvalidBody;
    nuka::scene::ShapeId shape = nuka::scene::kInvalidShape;
};

PathRecords ResolvePathRecords(const nuka::scene::SceneIR& s,
                               const std::string& path) {
    PathRecords out;
    const nuka::scene::SceneGraph& tree = s.Tree();
    const nuka::scene::Registry& ecs = s.Ecs();
    const auto target = tree.NodeOf(path);
    if (!target) return out;

    for (nuka::scene::BodyId i = 0; i < s.RigidBodyCount(); ++i) {
        const nuka::scene::EntityId e = s.EntityOfBody(i);
        if (e == nuka::scene::kInvalidEntity) continue;
        if (ecs.NodeOf(e) == target) {
            out.body = i;
            break;
        }
    }
    for (nuka::scene::ShapeId i = 0; i < s.ShapeCount(); ++i) {
        const nuka::scene::EntityId e = s.EntityOfShape(i);
        if (e == nuka::scene::kInvalidEntity) continue;
        if (ecs.NodeOf(e) == target) {
            out.shape = i;
            break;
        }
    }
    return out;
}

}  // namespace

extern "C" {

// ---------------------------------------------------------------------------
// load / destroy
// ---------------------------------------------------------------------------
nuka_result_t nuka_scene_load(const char* path, nuka_scene_handle* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out = nullptr;
    if (path == nullptr || path[0] == '\0') return NUKA_RESULT_INVALID_ARG;

    const std::string scene_path = path;
    if (!std::filesystem::exists(scene_path)) return NUKA_RESULT_FILE_NOT_FOUND;

    try {
        auto record = std::make_unique<SceneRecord>();
        record->scene = std::make_unique<nuka::scene::SceneIR>();
        if (!LoadByExtension(scene_path, record->scene.get())) {
            return NUKA_RESULT_INVALID_ARG;  // unrecognized extension.
        }
        *out = SceneTable().Insert(std::move(record));
        return (*out != nullptr) ? NUKA_RESULT_OK : NUKA_RESULT_INTERNAL;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

void nuka_scene_destroy(nuka_scene_handle scene) {
    (void)SceneTable().Remove(scene);
}

// ---------------------------------------------------------------------------
// compose
// ---------------------------------------------------------------------------
nuka_result_t nuka_scene_compose(nuka_scene_handle base, nuka_scene_handle addon,
                                 const float pos[3], const float quat[4],
                                 const char* attach_at) {
    SceneRecord* base_r = SceneTable().Get(base);
    SceneRecord* addon_r = SceneTable().Get(addon);
    if (base_r == nullptr || addon_r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!base_r->scene || !addon_r->scene) return NUKA_RESULT_NULL_HANDLE;
    try {
        const nuka::math::Transform placement = MakeTransform(pos, quat);
        const std::string prefix =
            (attach_at != nullptr) ? std::string(attach_at) : std::string();
        // Compose is PURE (returns by value); replace base's contents in place.
        nuka::scene::SceneIR merged =
            nuka::scene::Compose(*base_r->scene, *addon_r->scene, placement, prefix);
        *base_r->scene = std::move(merged);
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// find
// ---------------------------------------------------------------------------
nuka_result_t nuka_scene_find(nuka_scene_handle scene, const char* path,
                              int* out_found) {
    if (out_found == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out_found = 0;
    if (path == nullptr) return NUKA_RESULT_INVALID_ARG;
    SceneRecord* r = SceneTable().Get(scene);
    if (r == nullptr || !r->scene) return NUKA_RESULT_NULL_HANDLE;
    try {
        const auto node = r->scene->Tree().NodeOf(path);
        *out_found = (node != nullptr) ? 1 : 0;
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// set_local
// ---------------------------------------------------------------------------
nuka_result_t nuka_scene_set_local(nuka_scene_handle scene, const char* path,
                                   const float pos[3], const float quat[4]) {
    if (path == nullptr) return NUKA_RESULT_INVALID_ARG;
    SceneRecord* r = SceneTable().Get(scene);
    if (r == nullptr || !r->scene) return NUKA_RESULT_NULL_HANDLE;
    try {
        nuka::scene::SceneIR& s = *r->scene;
        const PathRecords ids = ResolvePathRecords(s, path);
        if (ids.body == nuka::scene::kInvalidBody &&
            ids.shape == nuka::scene::kInvalidShape) {
            return NUKA_RESULT_FILE_NOT_FOUND;  // no body/shape node at `path`.
        }
        // Merge into the existing local_transform: a NULL pos / quat leaves that
        // component as-authored. (GetBodyMut / GetShapeMut mark the facade dirty;
        // we resolved the records BEFORE mutating, so no stale handle survives.)
        if (ids.body != nuka::scene::kInvalidBody) {
            nuka::scene::RigidBodyRecord& rec = s.GetBodyMut(ids.body);
            if (pos != nullptr)
                rec.local_transform.position =
                    nuka::math::Vec3{pos[0], pos[1], pos[2]};
            if (quat != nullptr)
                rec.local_transform.rotation =
                    nuka::math::Quat{quat[0], quat[1], quat[2], quat[3]};
        } else {
            nuka::scene::CollisionShapeRecord& rec = s.GetShapeMut(ids.shape);
            if (pos != nullptr)
                rec.local_transform.position =
                    nuka::math::Vec3{pos[0], pos[1], pos[2]};
            if (quat != nullptr)
                rec.local_transform.rotation =
                    nuka::math::Quat{quat[0], quat[1], quat[2], quat[3]};
        }
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// add_collision_shape (attach a geom to an EXISTING body node by path)
// ---------------------------------------------------------------------------
nuka_result_t nuka_scene_add_collision_shape(
    nuka_scene_handle scene, const char* node_path,
    const nuka_collision_shape_desc_t* desc) {
    if (node_path == nullptr || desc == nullptr) return NUKA_RESULT_INVALID_ARG;
    SceneRecord* r = SceneTable().Get(scene);
    if (r == nullptr || !r->scene) return NUKA_RESULT_NULL_HANDLE;
    nuka::scene::ShapeType type;
    switch (desc->kind) {
        case NUKA_PRIMITIVE_BOX:     type = nuka::scene::ShapeType::Box; break;
        case NUKA_PRIMITIVE_SPHERE:  type = nuka::scene::ShapeType::Sphere; break;
        case NUKA_PRIMITIVE_CAPSULE: type = nuka::scene::ShapeType::Capsule; break;
        default: return NUKA_RESULT_INVALID_ARG;  // PLANE is not a body-attached geom.
    }
    if (type == nuka::scene::ShapeType::Sphere && desc->dims[0] <= 0.0f)
        return NUKA_RESULT_INVALID_ARG;
    if (type == nuka::scene::ShapeType::Capsule &&
        (desc->dims[0] <= 0.0f || desc->dims[1] <= 0.0f))
        return NUKA_RESULT_INVALID_ARG;
    if (type == nuka::scene::ShapeType::Box &&
        (desc->dims[0] <= 0.0f || desc->dims[1] <= 0.0f || desc->dims[2] <= 0.0f))
        return NUKA_RESULT_INVALID_ARG;
    try {
        nuka::scene::SceneIR& s = *r->scene;
        const PathRecords ids = ResolvePathRecords(s, node_path);
        if (ids.body == nuka::scene::kInvalidBody) {
            return NUKA_RESULT_FILE_NOT_FOUND;  // no body node at `node_path`.
        }
        nuka::scene::CollisionShapeRecord shape;
        shape.body_id = ids.body;
        shape.name = "added_shape_" + std::to_string(s.ShapeCount());
        shape.type = type;
        if (type == nuka::scene::ShapeType::Box) {
            shape.half_extents =
                nuka::math::Vec3{desc->dims[0], desc->dims[1], desc->dims[2]};
        } else if (type == nuka::scene::ShapeType::Sphere) {
            shape.radius = desc->dims[0];
        } else {
            shape.radius = desc->dims[0];
            shape.half_height = desc->dims[1];
        }
        // Local pose in the target body frame; an all-zero quat reads as identity.
        nuka::math::Transform local = nuka::math::Transform::Identity();
        local.position = nuka::math::Vec3{desc->pos[0], desc->pos[1], desc->pos[2]};
        const float qn = desc->quat[0] * desc->quat[0] + desc->quat[1] * desc->quat[1] +
                         desc->quat[2] * desc->quat[2] + desc->quat[3] * desc->quat[3];
        if (qn > 1.0e-12f) {
            local.rotation = nuka::math::Quat{desc->quat[0], desc->quat[1],
                                              desc->quat[2], desc->quat[3]};
        }
        shape.local_transform = local;
        if (desc->friction >= 0.0f) shape.friction_mu = desc->friction;
        shape.contype = desc->contype;
        shape.conaffinity = desc->conaffinity;
        s.AddCollisionShape(std::move(shape));
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// set_physics_material (regex over shape paths)
// ---------------------------------------------------------------------------
nuka_result_t nuka_scene_set_physics_material(nuka_scene_handle scene,
                                              const char* path_regex,
                                              float static_friction,
                                              float dynamic_friction,
                                              float restitution,
                                              uint32_t* out_matched) {
    if (out_matched != nullptr) *out_matched = 0u;
    if (path_regex == nullptr) return NUKA_RESULT_INVALID_ARG;
    SceneRecord* r = SceneTable().Get(scene);
    if (r == nullptr || !r->scene) return NUKA_RESULT_NULL_HANDLE;

    // The record carries a SINGLE isotropic friction (friction_mu); reject a
    // static != dynamic request rather than silently dropping one.
    if (static_friction >= 0.0f && dynamic_friction >= 0.0f &&
        static_friction != dynamic_friction) {
        return NUKA_RESULT_INVALID_ARG;
    }
    const float mu =
        (static_friction >= 0.0f) ? static_friction : dynamic_friction;  // <0 == leave

    try {
        std::regex re;
        try {
            re = std::regex(path_regex);
        } catch (const std::regex_error&) {
            return NUKA_RESULT_INVALID_ARG;  // malformed regex.
        }

        nuka::scene::SceneIR& s = *r->scene;
        // Snapshot the (shape_id, derived path) pairs from the CURRENT facade
        // BEFORE any mutation -- GetShapeMut would re-project the facade and
        // invalidate the entity/node handles mid-loop.
        std::vector<nuka::scene::ShapeId> matched;
        {
            const nuka::scene::SceneGraph& tree = s.Tree();
            const nuka::scene::Registry& ecs = s.Ecs();
            for (nuka::scene::ShapeId i = 0; i < s.ShapeCount(); ++i) {
                const nuka::scene::EntityId e = s.EntityOfShape(i);
                if (e == nuka::scene::kInvalidEntity) continue;
                const auto node = ecs.NodeOf(e);
                if (!node) continue;
                const std::string path = tree.PathOf(node);
                if (std::regex_match(path, re)) matched.push_back(i);
            }
        }

        // Apply: write the persisted per-shape friction_mu (mu >= 0). restitution
        // has NO record/.nks field, so we also patch the matched shapes' RESOLVED
        // PhysicsMaterial in the live facade Registry for in-memory cook consumers
        // (documented in nuka_scene.h: restitution does NOT survive Save). The
        // friction write goes LAST per shape so the facade settles on the records.
        for (nuka::scene::ShapeId i : matched) {
            if (restitution >= 0.0f) {
                // Resolve the shape's PhysicsMaterial id from the fresh facade and
                // set its restitution (best-effort; does not persist).
                const nuka::scene::Registry& cecs = s.Ecs();
                const nuka::scene::EntityId e = s.EntityOfShape(i);
                if (e != nuka::scene::kInvalidEntity) {
                    if (const auto* csc =
                            cecs.Get<nuka::scene::CollisionShapeComponent>(e)) {
                        const uint32_t pid = csc->physics_material_id;
                        // Const-cast onto the facade's material asset table: this
                        // is an in-memory-only side channel (records are the Save
                        // authority; restitution has no record field).
                        auto& mecs =
                            const_cast<nuka::scene::Registry&>(cecs);
                        if (nuka::scene::PhysicsMaterial* pm =
                                mecs.GetPhysicsMaterial(pid)) {
                            pm->restitution = restitution;
                        }
                    }
                }
            }
            if (mu >= 0.0f) {
                nuka::scene::CollisionShapeRecord& rec = s.GetShapeMut(i);
                rec.friction_mu = mu;  // persisted (static==dynamic, isotropic).
            }
        }

        if (out_matched != nullptr)
            *out_matched = static_cast<uint32_t>(matched.size());
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// settle (device; CUDA-gated)
// ---------------------------------------------------------------------------
nuka_result_t nuka_scene_settle(nuka_scene_handle scene,
                                nuka_device_handle device, uint32_t steps,
                                float dt) {
    SceneRecord* r = SceneTable().Get(scene);
    if (r == nullptr || !r->scene) return NUKA_RESULT_NULL_HANDLE;
    DeviceRecord* device_record = nuka::c_abi::DeviceTable().Get(device);
    if (device_record == nullptr) return NUKA_RESULT_NULL_HANDLE;

    // CUDA-less build / no phi v2 backend acquired -> settle is unavailable.
    if (device_record->phi_device == nullptr ||
        device_record->backend == nullptr) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }

    try {
        nuka::scene::SceneIR& s = *r->scene;
        const float use_dt = (dt > 0.0f) ? dt : (1.0f / 240.0f);

        // Cook the authored scene to an nk::World on the device's OWNED backend
        // (mirroring recorder.cpp's Load -> CookToModel -> nk::World seam).
        cook::CookToModelResult cooked = cook::CookToModel(s, 1);
        nk::Pipeline::SolverConfig cfg;
        cfg.dt = use_dt;
        cfg.gravity[0] = 0.0f;
        cfg.gravity[1] = 0.0f;
        cfg.gravity[2] = -9.81f;
        nk::World world(std::move(cooked.model), 1u, device_record->phi_device,
                        device_record->backend, cfg);
        if (!world.Ready()) return NUKA_RESULT_NOT_SUPPORTED;

        // The settle directive: the scene's AUTHORED Settle() spec if it carries
        // holds, else a generic "hold every articulation link at its seeded q"
        // default (path glob "*", a firm generic PD hold). dt/steps from the args
        // (steps arg wins over any authored steps so the caller controls duration).
        cook::SettleSpec spec = s.Settle();
        if (spec.holds.empty()) {
            cook::SettleSpec::Hold hold;
            hold.dof_pattern = "*";
            hold.mode = cook::SettleSpec::Hold::Mode::PD;
            hold.kp = 50.0f;
            hold.kd = 4.0f;
            spec.holds.push_back(hold);
        }
        spec.steps = static_cast<int>(steps);
        spec.dt = use_dt;

        const cook::SettleResult result =
            cook::Settle(world, s, cooked.scene_map, spec);
        if (!result.ok) return NUKA_RESULT_NOT_SUPPORTED;

        // Writeback (movable FREE-body record poses -> persisted by Save).
        cook::ApplySettleToSceneIR(s, result, cooked.scene_map);

        // Writeback the settled articulation IC into the persisted
        // SceneInitialState (keyed by the cooked articulation root's DERIVED node
        // path -- e.g. "h1"). The root link's entity is link row 0; close the
        // loop entity -> node -> path AFTER ApplySettleToSceneIR's record edits
        // (re-read the fresh facade). If the root path cannot be resolved (no
        // articulation / unbound), the IC is still discoverable on the records'
        // body local_transforms; we skip the IC key rather than guess.
        if (!result.initial_state.qpos.empty()) {
            const nuka::scene::EntityId root_link =
                cooked.scene_map.EntityOfLink(0u);
            std::string key;
            if (root_link != nuka::scene::kInvalidEntity) {
                const auto node = s.Ecs().NodeOf(root_link);
                if (node) key = s.Tree().PathOf(node);
            }
            if (key.empty()) {
                // Fallback: the articulation root body row (first body whose
                // parent is invalid), so a non-link-mapped cook still keys an IC.
                for (nuka::scene::BodyId b = 0; b < s.RigidBodyCount(); ++b) {
                    if (s.GetBody(b).parent_id == nuka::scene::kInvalidBody) {
                        const nuka::scene::EntityId e = s.EntityOfBody(b);
                        const auto node =
                            (e != nuka::scene::kInvalidEntity) ? s.Ecs().NodeOf(e)
                                                               : nullptr;
                        if (node) {
                            key = s.Tree().PathOf(node);
                            // Use only the FIRST path segment (the articulation
                            // import prefix, e.g. "h1") to match the authored
                            // initial_state key convention.
                            const auto slash = key.find('/');
                            if (slash != std::string::npos) key.resize(slash);
                        }
                        break;
                    }
                }
            }
            if (!key.empty()) {
                nuka::scene::ArticulationInitialState ais;
                ais.qpos = result.initial_state.qpos;
                ais.root = result.initial_state.root;
                s.InitialStateMut()[key] = std::move(ais);
            }
        }

        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// save
// ---------------------------------------------------------------------------
nuka_result_t nuka_scene_save(nuka_scene_handle scene, const char* nks_path) {
    if (nks_path == nullptr || nks_path[0] == '\0') return NUKA_RESULT_INVALID_ARG;
    SceneRecord* r = SceneTable().Get(scene);
    if (r == nullptr || !r->scene) return NUKA_RESULT_NULL_HANDLE;
    try {
        nks::Save(*r->scene, nks_path);
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

}  // extern "C"
