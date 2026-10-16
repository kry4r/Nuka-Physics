// The tunable parameter registry of a world: world options and per-template-link model
// values. A LIVE write also lands in the retained scene, so export and rebuild keep it.

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "nk/pipeline/world.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "scene/scene_fields.hpp"
#include "scene/scene_ir.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <new>

namespace {

namespace nk = nuka::nk;
namespace articulation = nuka::runtime::articulation;
using nuka::c_abi::WorldRecord;
using nuka::phi::Status;

enum class Param : uint32_t {
    GravityZ,
    VelocityIterations,
    LinkMass,
    JointDamping,
    JointArmature,
    JointFrictionloss,
    DriveStiffness,
    DriveForceLimit
};

struct ParamDef {
    const char* key;
    const char* unit;
    const char* scene_field;  // empty for world options
    nuka_param_type_t type;
    double min_value;
    double max_value;
};

constexpr double kFloatMin = 1.1754943508222875e-38;
constexpr double kFloatMax = 3.4028234663852886e38;

constexpr ParamDef kParams[] = {
    {"gravity_z", "m/s^2", "", NUKA_PARAM_F32, -1000.0, 1000.0},
    {"velocity_iterations", "", "", NUKA_PARAM_U32, 1.0, 65535.0},
    {"link_mass", "kg", "body.mass", NUKA_PARAM_F32, kFloatMin, kFloatMax},
    {"joint_damping", "N*m*s/rad", "joint.damping", NUKA_PARAM_F32, 0.0, kFloatMax},
    {"joint_armature", "kg*m^2", "joint.armature", NUKA_PARAM_F32, 0.0, kFloatMax},
    {"joint_frictionloss", "N*m", "joint.frictionloss", NUKA_PARAM_F32, 0.0, kFloatMax},
    {"drive_stiffness", "N*m/rad", "actuator.gain", NUKA_PARAM_F32, 0.0, kFloatMax},
    {"drive_force_limit", "N*m", "actuator.force_limit", NUKA_PARAM_F32, 0.0, kFloatMax},
};
constexpr uint32_t kParamCount = static_cast<uint32_t>(sizeof(kParams) / sizeof(kParams[0]));

const ParamDef& Def(Param p) { return kParams[static_cast<uint32_t>(p)]; }

bool PerLink(Param p) { return p != Param::GravityZ && p != Param::VelocityIterations; }

uint32_t Count(const WorldRecord& record, Param p) {
    return PerLink(p) ? record.world->GetModel().capacities.links_per_env : 1u;
}

// Joint friction is solved in rows the cook allocates only when some joint authors it.
nuka_param_tier_t Tier(const WorldRecord& record, Param p) {
    return p == Param::JointFrictionloss &&
                   record.world->GetModel().capacities.joint_friction_rows_per_env == 0u
               ? NUKA_PARAM_REBUILD
               : NUKA_PARAM_LIVE;
}

double Value(const WorldRecord& record, Param p, uint32_t link) {
    const nk::World& world = *record.world;
    const nk::ModelArticulation& art = world.GetModel().articulation;
    const nk::ModelHoldDrives& drives = world.GetModel().hold_drives;
    constexpr size_t kMassEntry = 3u * 6u + 3u;  // (3,3) of the row-major 6x6 spatial inertia
    switch (p) {
        case Param::GravityZ: return world.Gravity().z;
        case Param::VelocityIterations: return world.VelocityIterations();
        case Param::LinkMass: return art.link_inertia_spatial[size_t(link) * 36u + kMassEntry];
        case Param::JointDamping: return art.joint_damping[link];
        case Param::JointArmature: return art.joint_armature[link];
        case Param::JointFrictionloss: return art.joint_frictionloss[link];
        case Param::DriveStiffness: return drives.stiffness[link];
        case Param::DriveForceLimit: return drives.force_limits[link];
    }
    return 0.0;
}

articulation::ArticulationCookedTopology* TopologyOf(articulation::ArticulationHostState& host,
                                                     uint32_t link, uint32_t* local) {
    uint32_t first = 0u;
    for (auto& topo : host.articulations) {
        const auto count = static_cast<uint32_t>(topo.link_bodies.size());
        if (link < first + count) {
            *local = link - first;
            return &topo;
        }
        first += count;
    }
    return nullptr;
}

// Writes one link value to the world and the host mirrors checkpoints read.
Status SetLink(WorldRecord& record, Param p, uint32_t link, float value) {
    nk::World& world = *record.world;
    auto& host = record.articulation_host;
    uint32_t local = 0u;
    articulation::ArticulationCookedTopology& topo = *TopologyOf(host, link, &local);
    Status status = Status::Ok;
    switch (p) {
        case Param::LinkMass: {
            // Mass changes with the COM and the rotational inertia at the COM held fixed.
            const articulation::LinkSpatialInertia inertia =
                articulation::MakeSpatialInertia(value, topo.inertias[local], topo.inertial_frames[local]);
            status = world.SetLinkParam(nk::FieldId::LinkInertia, link, inertia.I, 36u);
            if (status != Status::Ok) return status;
            host.link_inertia[link] = inertia;
            topo.masses[local] = value;
            return status;
        }
        case Param::JointDamping:
            status = world.SetLinkParam(nk::FieldId::JointDamping, link, &value, 1u);
            if (status == Status::Ok) host.joint_damping[link] = topo.joint_dampings[local] = value;
            return status;
        case Param::JointArmature:
            status = world.SetLinkParam(nk::FieldId::JointArmature, link, &value, 1u);
            if (status == Status::Ok) host.joint_armature[link] = topo.joint_armatures[local] = value;
            return status;
        case Param::JointFrictionloss:
            status = world.SetLinkParam(nk::FieldId::JointFrictionloss, link, &value, 1u);
            if (status == Status::Ok) host.joint_frictionloss[link] = topo.joint_frictionlosses[local] = value;
            return status;
        case Param::DriveStiffness: {
            // The cook derives a position drive's damping from its gain the same way.
            const float damping = 2.0f * std::sqrt(value);
            status = world.SetLinkParam(nk::FieldId::DriveStiffness, link, &value, 1u);
            if (status == Status::Ok) status = world.SetLinkParam(nk::FieldId::DriveDamping, link, &damping, 1u);
            return status;
        }
        case Param::DriveForceLimit:
            return world.SetLinkParam(nk::FieldId::DriveForceLimit, link, &value, 1u);
        case Param::GravityZ:
        case Param::VelocityIterations:
            break;
    }
    return Status::InvalidArgument;
}

nuka_result_t SetParam(WorldRecord& record, Param p, uint32_t element, double value) {
    const ParamDef& def = Def(p);
    if (element >= Count(record, p) || !std::isfinite(value) || value < def.min_value ||
        value > def.max_value || (def.type == NUKA_PARAM_U32 && value != std::floor(value)))
        return NUKA_RESULT_INVALID_ARG;
    if (Tier(record, p) != NUKA_PARAM_LIVE) return NUKA_RESULT_NOT_SUPPORTED;
    nk::World& world = *record.world;
    const auto f = static_cast<float>(value);
    if (p == Param::GravityZ) {
        nuka::math::Vec3 gravity = world.Gravity();
        gravity.z = f;
        const Status status = world.SetGravity(gravity);
        if (status == Status::Ok) record.step_options.gravity = gravity;
        return nuka::c_abi::MapStatusToResult(status);
    }
    if (p == Param::VelocityIterations) {
        const Status status = world.SetVelocityIterations(static_cast<uint32_t>(value));
        if (status == Status::Ok) record.velocity_iterations_set = true;
        return nuka::c_abi::MapStatusToResult(status);
    }
    nuka::scene::SceneTarget target;
    target.body = world.GetModel().articulation.link_body[element];
    float authored = 0.0f;
    if (!nuka::scene::GetSceneField(*record.scene, target, def.scene_field, &authored))
        return NUKA_RESULT_NOT_SUPPORTED;
    const Status status = SetLink(record, p, element, f);
    if (status != Status::Ok) return nuka::c_abi::MapStatusToResult(status);
    nuka::scene::SetSceneField(*record.scene, target, def.scene_field, f);
    return NUKA_RESULT_OK;
}

}  // namespace

extern "C" {

nuka_result_t nuka_world_param_count(nuka_world_handle world, uint32_t* out_count) {
    if (out_count == nullptr) return NUKA_RESULT_INVALID_ARG;
    if (nuka::c_abi::WorldTable().Get(world) == nullptr) return NUKA_RESULT_NULL_HANDLE;
    *out_count = kParamCount;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_get_param_infos(nuka_world_handle world, nuka_param_info_t* infos,
                                         uint32_t capacity) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    if (infos == nullptr || capacity < kParamCount) return NUKA_RESULT_INVALID_ARG;
    for (uint32_t i = 0u; i < kParamCount; ++i)
        if (infos[i].struct_size != sizeof(nuka_param_info_t) ||
            infos[i].schema_version != NUKA_PARAM_INFO_VERSION)
            return NUKA_RESULT_INVALID_ARG;
    for (uint32_t i = 0u; i < kParamCount; ++i) {
        const auto p = static_cast<Param>(i);
        const ParamDef& def = Def(p);
        nuka_param_info_t& info = infos[i];
        info = {};
        info.struct_size = sizeof(info);
        info.schema_version = NUKA_PARAM_INFO_VERSION;
        std::snprintf(info.key, sizeof(info.key), "%s", def.key);
        std::snprintf(info.unit, sizeof(info.unit), "%s", def.unit);
        std::snprintf(info.scene_field, sizeof(info.scene_field), "%s", def.scene_field);
        info.type = def.type;
        info.tier = Tier(*record, p);
        info.count = Count(*record, p);
        info.min_value = def.min_value;
        info.max_value = def.max_value;
    }
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_get_param(nuka_world_handle world, uint32_t param, uint32_t element,
                                   double* out_value) {
    if (out_value == nullptr) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    if (param >= kParamCount || element >= Count(*record, static_cast<Param>(param)))
        return NUKA_RESULT_INVALID_ARG;
    *out_value = Value(*record, static_cast<Param>(param), element);
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_set_param(nuka_world_handle world, uint32_t param, uint32_t element,
                                   double value) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    if (param >= kParamCount) return NUKA_RESULT_INVALID_ARG;
    try {
        return SetParam(*record, static_cast<Param>(param), element, value);
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_set_link_mass(nuka_world_handle world, uint32_t link_index, float mass) {
    return nuka_world_set_param(world, static_cast<uint32_t>(Param::LinkMass), link_index, mass);
}

nuka_result_t nuka_world_set_gravity_z(nuka_world_handle world, float gravity_z) {
    return nuka_world_set_param(world, static_cast<uint32_t>(Param::GravityZ), 0u, gravity_z);
}

nuka_result_t nuka_world_set_velocity_iterations(nuka_world_handle world, uint32_t iterations) {
    return nuka_world_set_param(world, static_cast<uint32_t>(Param::VelocityIterations), 0u, iterations);
}

}  // extern "C"
