#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::Registry - explicit, closed-set ECS registry.
//
// Generational entity slots (index + gen) over a free-list; one dense pool per
// component type with a per-type sparse index (entity-slot -> dense-slot). The
// component set is fixed (components.hpp), so the pools are explicit members
// rather than generic archetype machinery -- readable over clever.
//
// Materials are id-referenced ASSETS, not per-entity components, so they live
// in standalone asset tables (physics_materials_ / render_materials_) with a
// name->id map. Each entity also carries a weak backref to its scene node
// (BindNode / NodeOf); path addressing lives on SceneGraph, never here.
// ---------------------------------------------------------------------------

#include "scene/ecs/components.hpp"
#include "scene/ecs/entity.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace nuka::scene {

struct SceneNode;  // scene/graph/scene_graph.hpp (node backref only)

class Registry {
public:
    static constexpr uint32_t kNoSlot = ~uint32_t(0);

    // -- entity lifecycle ---------------------------------------------------
    EntityId Create();
    void     Destroy(EntityId e);   // bumps gen, releases all components + backref
    bool     Alive(EntityId e) const;
    size_t   AliveCount() const { return alive_count_; }

    // -- per-component access (one explicit overload set per type) ----------
    // Add returns a reference to the stored component (overwrites if present).
    NameComponent&            Add(EntityId e, NameComponent c);
    TransformComponent&       Add(EntityId e, TransformComponent c);
    VisualMeshComponent&      Add(EntityId e, VisualMeshComponent c);
    CollisionShapeComponent&  Add(EntityId e, CollisionShapeComponent c);
    RigidBodyComponent&       Add(EntityId e, RigidBodyComponent c);
    JointComponent&           Add(EntityId e, JointComponent c);
    ActuatorComponent&        Add(EntityId e, ActuatorComponent c);
    SystemKindComponent&      Add(EntityId e, SystemKindComponent c);
    SoftBodyComponent&        Add(EntityId e, SoftBodyComponent c);
    FluidComponent&           Add(EntityId e, FluidComponent c);
    InitialStateComponent&    Add(EntityId e, InitialStateComponent c);
    CameraComponent&          Add(EntityId e, CameraComponent c);
    LightComponent&           Add(EntityId e, LightComponent c);

    // Get<T>(e) -> T* (null if the entity is dead or lacks the component).
    template <class T>       T* Get(EntityId e);
    template <class T> const T* Get(EntityId e) const;
    template <class T> bool     Has(EntityId e) const { return const_cast<Registry*>(this)->Get<T>(e) != nullptr; }
    template <class T> bool     Remove(EntityId e);

    // ForEach<T>(fn(EntityId, T&)) over the dense pool in stable slot order.
    template <class T, class F> void ForEach(F&& fn);
    template <class T, class F> void ForEach(F&& fn) const;

    // -- material asset tables ----------------------------------------------
    uint32_t                       AddPhysicsMaterial(const std::string& name, PhysicsMaterial m);
    uint32_t                       AddRenderMaterial(const std::string& name, RenderMaterial m);
    PhysicsMaterial*               GetPhysicsMaterial(uint32_t id);
    const PhysicsMaterial*         GetPhysicsMaterial(uint32_t id) const;
    RenderMaterial*                GetRenderMaterial(uint32_t id);
    const RenderMaterial*          GetRenderMaterial(uint32_t id) const;
    uint32_t                       FindPhysicsMaterial(const std::string& name) const;  // kNoSlot if absent
    uint32_t                       FindRenderMaterial(const std::string& name) const;   // kNoSlot if absent
    size_t                         PhysicsMaterialCount() const { return physics_materials_.size(); }
    size_t                         RenderMaterialCount() const { return render_materials_.size(); }

    // -- entity -> scene-node backref (weak) --------------------------------
    void                        BindNode(EntityId e, std::shared_ptr<SceneNode> node);
    std::shared_ptr<SceneNode>  NodeOf(EntityId e) const;

private:
    // One dense pool per component type. `dense` holds the components, `owners`
    // holds the EntityId for each dense slot, and `sparse` maps an entity slot
    // index to its dense slot (kNoSlot if absent). Swap-and-pop keeps `dense`
    // contiguous; `sparse` is patched for the entity that moved into the hole.
    template <class T>
    struct Pool {
        std::vector<T>        dense;
        std::vector<EntityId> owners;
        std::vector<uint32_t> sparse;  // entity index -> dense slot

        T&   Add(EntityId e, T value);
        T*   Get(EntityId e);
        bool Remove(EntityId e);
        void EnsureSparse(uint32_t index);
    };

    bool ValidSlot(EntityId e) const;

    // entity slot metadata
    std::vector<uint32_t> gens_;       // per-slot generation counter
    std::vector<uint32_t> free_list_;  // recycled slot indices
    size_t                alive_count_ = 0;

    // backref storage (entity index -> weak node), grown lazily
    std::vector<std::weak_ptr<SceneNode>> node_refs_;

    // component pools
    Pool<NameComponent>           names_;
    Pool<TransformComponent>      transforms_;
    Pool<VisualMeshComponent>     visual_meshes_;
    Pool<CollisionShapeComponent> collision_shapes_;
    Pool<RigidBodyComponent>      rigid_bodies_;
    Pool<JointComponent>          joints_;
    Pool<ActuatorComponent>       actuators_;
    Pool<SystemKindComponent>     system_kinds_;
    Pool<SoftBodyComponent>       soft_bodies_;
    Pool<FluidComponent>          fluids_;
    Pool<InitialStateComponent>   initial_states_;
    Pool<CameraComponent>         cameras_;
    Pool<LightComponent>          lights_;

    // material asset tables
    std::vector<PhysicsMaterial>                  physics_materials_;
    std::vector<RenderMaterial>                   render_materials_;
    std::unordered_map<std::string, uint32_t>     physics_material_ids_;
    std::unordered_map<std::string, uint32_t>     render_material_ids_;

    // -- pool accessor: maps a component type to its member pool -------------
    template <class T> Pool<T>&       PoolFor();
    template <class T> const Pool<T>& PoolFor() const {
        return const_cast<Registry*>(this)->PoolFor<T>();
    }
};

// ---------------------------------------------------------------------------
// Pool<T> implementation
// ---------------------------------------------------------------------------

template <class T>
void Registry::Pool<T>::EnsureSparse(uint32_t index) {
    if (index >= sparse.size()) {
        sparse.resize(index + 1, Registry::kNoSlot);
    }
}

template <class T>
T& Registry::Pool<T>::Add(EntityId e, T value) {
    EnsureSparse(e.index);
    const uint32_t existing = sparse[e.index];
    if (existing != Registry::kNoSlot) {
        dense[existing]  = std::move(value);
        owners[existing] = e;
        return dense[existing];
    }
    const auto slot = static_cast<uint32_t>(dense.size());
    dense.push_back(std::move(value));
    owners.push_back(e);
    sparse[e.index] = slot;
    return dense[slot];
}

template <class T>
T* Registry::Pool<T>::Get(EntityId e) {
    if (e.index >= sparse.size()) {
        return nullptr;
    }
    const uint32_t slot = sparse[e.index];
    if (slot == Registry::kNoSlot) {
        return nullptr;
    }
    // Guard against a stale handle pointing at a recycled slot.
    if (owners[slot] != e) {
        return nullptr;
    }
    return &dense[slot];
}

template <class T>
bool Registry::Pool<T>::Remove(EntityId e) {
    if (e.index >= sparse.size()) {
        return false;
    }
    const uint32_t slot = sparse[e.index];
    if (slot == Registry::kNoSlot) {
        return false;
    }
    const auto last = static_cast<uint32_t>(dense.size() - 1);
    if (slot != last) {
        // swap the last element into the hole and repoint its sparse entry
        dense[slot]  = std::move(dense[last]);
        owners[slot] = owners[last];
        sparse[owners[slot].index] = slot;
    }
    dense.pop_back();
    owners.pop_back();
    sparse[e.index] = Registry::kNoSlot;
    return true;
}

// ---------------------------------------------------------------------------
// Registry templated members
// ---------------------------------------------------------------------------

template <class T>
T* Registry::Get(EntityId e) {
    if (!Alive(e)) {
        return nullptr;
    }
    return PoolFor<T>().Get(e);
}

template <class T>
const T* Registry::Get(EntityId e) const {
    if (!Alive(e)) {
        return nullptr;
    }
    return const_cast<Registry*>(this)->PoolFor<T>().Get(e);
}

template <class T>
bool Registry::Remove(EntityId e) {
    if (!ValidSlot(e)) {
        return false;
    }
    return PoolFor<T>().Remove(e);
}

template <class T, class F>
void Registry::ForEach(F&& fn) {
    auto& pool = PoolFor<T>();
    for (size_t i = 0; i < pool.dense.size(); ++i) {
        fn(pool.owners[i], pool.dense[i]);
    }
}

template <class T, class F>
void Registry::ForEach(F&& fn) const {
    const auto& pool = PoolFor<T>();
    for (size_t i = 0; i < pool.dense.size(); ++i) {
        fn(pool.owners[i], pool.dense[i]);
    }
}

// Explicit type->pool mapping. A compile error here means a component type was
// used with Get/Remove/ForEach that has no pool member -- add one above.
template <> inline Registry::Pool<NameComponent>&           Registry::PoolFor<NameComponent>()           { return names_; }
template <> inline Registry::Pool<TransformComponent>&      Registry::PoolFor<TransformComponent>()      { return transforms_; }
template <> inline Registry::Pool<VisualMeshComponent>&     Registry::PoolFor<VisualMeshComponent>()     { return visual_meshes_; }
template <> inline Registry::Pool<CollisionShapeComponent>& Registry::PoolFor<CollisionShapeComponent>() { return collision_shapes_; }
template <> inline Registry::Pool<RigidBodyComponent>&      Registry::PoolFor<RigidBodyComponent>()      { return rigid_bodies_; }
template <> inline Registry::Pool<JointComponent>&          Registry::PoolFor<JointComponent>()          { return joints_; }
template <> inline Registry::Pool<ActuatorComponent>&       Registry::PoolFor<ActuatorComponent>()       { return actuators_; }
template <> inline Registry::Pool<SystemKindComponent>&     Registry::PoolFor<SystemKindComponent>()     { return system_kinds_; }
template <> inline Registry::Pool<SoftBodyComponent>&       Registry::PoolFor<SoftBodyComponent>()       { return soft_bodies_; }
template <> inline Registry::Pool<FluidComponent>&          Registry::PoolFor<FluidComponent>()          { return fluids_; }
template <> inline Registry::Pool<InitialStateComponent>&   Registry::PoolFor<InitialStateComponent>()   { return initial_states_; }
template <> inline Registry::Pool<CameraComponent>&         Registry::PoolFor<CameraComponent>()         { return cameras_; }
template <> inline Registry::Pool<LightComponent>&          Registry::PoolFor<LightComponent>()          { return lights_; }

} // namespace nuka::scene
