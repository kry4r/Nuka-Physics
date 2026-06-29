// ---------------------------------------------------------------------------
// nuka::scene::Registry - non-template members.
// ---------------------------------------------------------------------------

#include "scene/ecs/registry.hpp"

#include <cassert>
#include <utility>

namespace nuka::scene {

// Generational guard for Add: a STALE handle (whose slot was recycled) would
// otherwise overwrite the new occupant's component and re-point the pool's
// owner row at the dead handle (silent cross-entity corruption). Adding to a
// dead/stale entity is a caller bug; the contract is asserted here (Add must
// return a reference, so there is no error channel — the guard makes the
// misuse loud instead of silently corrupting).
#define NUKA_REGISTRY_ADD_GUARD(e)                                       \
    assert(ValidSlot(e) && "Registry::Add on a dead/stale EntityId")

// -- entity lifecycle -------------------------------------------------------

EntityId Registry::Create() {
    EntityId e;
    if (!free_list_.empty()) {
        e.index = free_list_.back();
        free_list_.pop_back();
        e.gen = gens_[e.index];
    } else {
        e.index = static_cast<uint32_t>(gens_.size());
        gens_.push_back(0);
        e.gen = 0;
    }
    ++alive_count_;
    return e;
}

bool Registry::ValidSlot(EntityId e) const {
    return e.index < gens_.size() && gens_[e.index] == e.gen;
}

bool Registry::Alive(EntityId e) const {
    // A live entity has a matching generation AND is not currently on the
    // free-list. Generation match alone is enough because Destroy bumps the
    // generation, so a freed slot's stored gen differs from any handle that was
    // valid before the free (and a recycled slot gets a fresh handle with the
    // new gen).
    return ValidSlot(e);
}

void Registry::Destroy(EntityId e) {
    if (!ValidSlot(e)) {
        return;
    }
    // Release every component pool for this entity.
    names_.Remove(e);
    transforms_.Remove(e);
    visual_meshes_.Remove(e);
    collision_shapes_.Remove(e);
    rigid_bodies_.Remove(e);
    joints_.Remove(e);
    actuators_.Remove(e);
    system_kinds_.Remove(e);
    soft_bodies_.Remove(e);
    fluids_.Remove(e);
    initial_states_.Remove(e);
    cameras_.Remove(e);
    lights_.Remove(e);
    scripts_.Remove(e);

    // Drop the node backref.
    if (e.index < node_refs_.size()) {
        node_refs_[e.index].reset();
    }

    // Bump generation so stale handles compare unequal, then recycle the slot.
    ++gens_[e.index];
    free_list_.push_back(e.index);
    --alive_count_;
}

// -- per-component Add overloads --------------------------------------------

NameComponent&           Registry::Add(EntityId e, NameComponent c)           { NUKA_REGISTRY_ADD_GUARD(e); return names_.Add(e, std::move(c)); }
TransformComponent&      Registry::Add(EntityId e, TransformComponent c)      { NUKA_REGISTRY_ADD_GUARD(e); return transforms_.Add(e, std::move(c)); }
VisualMeshComponent&     Registry::Add(EntityId e, VisualMeshComponent c)     { NUKA_REGISTRY_ADD_GUARD(e); return visual_meshes_.Add(e, std::move(c)); }
CollisionShapeComponent& Registry::Add(EntityId e, CollisionShapeComponent c) { NUKA_REGISTRY_ADD_GUARD(e); return collision_shapes_.Add(e, std::move(c)); }
RigidBodyComponent&      Registry::Add(EntityId e, RigidBodyComponent c)      { NUKA_REGISTRY_ADD_GUARD(e); return rigid_bodies_.Add(e, std::move(c)); }
JointComponent&          Registry::Add(EntityId e, JointComponent c)          { NUKA_REGISTRY_ADD_GUARD(e); return joints_.Add(e, std::move(c)); }
ActuatorComponent&       Registry::Add(EntityId e, ActuatorComponent c)       { NUKA_REGISTRY_ADD_GUARD(e); return actuators_.Add(e, std::move(c)); }
SystemKindComponent&     Registry::Add(EntityId e, SystemKindComponent c)     { NUKA_REGISTRY_ADD_GUARD(e); return system_kinds_.Add(e, std::move(c)); }
SoftBodyComponent&       Registry::Add(EntityId e, SoftBodyComponent c)       { NUKA_REGISTRY_ADD_GUARD(e); return soft_bodies_.Add(e, std::move(c)); }
FluidComponent&          Registry::Add(EntityId e, FluidComponent c)          { NUKA_REGISTRY_ADD_GUARD(e); return fluids_.Add(e, std::move(c)); }
InitialStateComponent&   Registry::Add(EntityId e, InitialStateComponent c)   { NUKA_REGISTRY_ADD_GUARD(e); return initial_states_.Add(e, std::move(c)); }
CameraComponent&         Registry::Add(EntityId e, CameraComponent c)         { NUKA_REGISTRY_ADD_GUARD(e); return cameras_.Add(e, std::move(c)); }
LightComponent&          Registry::Add(EntityId e, LightComponent c)          { NUKA_REGISTRY_ADD_GUARD(e); return lights_.Add(e, std::move(c)); }
ScriptComponent&         Registry::Add(EntityId e, ScriptComponent c)         { NUKA_REGISTRY_ADD_GUARD(e); return scripts_.Add(e, std::move(c)); }

// -- material asset tables --------------------------------------------------

uint32_t Registry::AddPhysicsMaterial(const std::string& name, PhysicsMaterial m) {
    const auto it = physics_material_ids_.find(name);
    if (it != physics_material_ids_.end()) {
        physics_materials_[it->second] = m;  // overwrite existing by name
        return it->second;
    }
    const auto id = static_cast<uint32_t>(physics_materials_.size());
    physics_materials_.push_back(m);
    physics_material_ids_.emplace(name, id);
    return id;
}

uint32_t Registry::AddRenderMaterial(const std::string& name, RenderMaterial m) {
    const auto it = render_material_ids_.find(name);
    if (it != render_material_ids_.end()) {
        render_materials_[it->second] = std::move(m);
        return it->second;
    }
    const auto id = static_cast<uint32_t>(render_materials_.size());
    render_materials_.push_back(std::move(m));
    render_material_ids_.emplace(name, id);
    return id;
}

PhysicsMaterial* Registry::GetPhysicsMaterial(uint32_t id) {
    return id < physics_materials_.size() ? &physics_materials_[id] : nullptr;
}
const PhysicsMaterial* Registry::GetPhysicsMaterial(uint32_t id) const {
    return id < physics_materials_.size() ? &physics_materials_[id] : nullptr;
}
RenderMaterial* Registry::GetRenderMaterial(uint32_t id) {
    return id < render_materials_.size() ? &render_materials_[id] : nullptr;
}
const RenderMaterial* Registry::GetRenderMaterial(uint32_t id) const {
    return id < render_materials_.size() ? &render_materials_[id] : nullptr;
}

uint32_t Registry::FindPhysicsMaterial(const std::string& name) const {
    const auto it = physics_material_ids_.find(name);
    return it != physics_material_ids_.end() ? it->second : kNoSlot;
}
uint32_t Registry::FindRenderMaterial(const std::string& name) const {
    const auto it = render_material_ids_.find(name);
    return it != render_material_ids_.end() ? it->second : kNoSlot;
}

// -- entity -> node backref -------------------------------------------------

void Registry::BindNode(EntityId e, std::shared_ptr<SceneNode> node) {
    if (!ValidSlot(e)) {
        return;
    }
    if (e.index >= node_refs_.size()) {
        node_refs_.resize(e.index + 1);
    }
    node_refs_[e.index] = std::move(node);
}

std::shared_ptr<SceneNode> Registry::NodeOf(EntityId e) const {
    if (!ValidSlot(e) || e.index >= node_refs_.size()) {
        return nullptr;
    }
    return node_refs_[e.index].lock();
}

} // namespace nuka::scene
