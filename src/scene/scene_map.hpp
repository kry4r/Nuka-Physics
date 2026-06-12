#pragma once
// ---------------------------------------------------------------------------
// nuka::scene::SceneMap - bidirectional EntityId <-> cooked-row mapping.
//
// The cook (M3/M7) flattens the ECS scene into row-indexed device arrays
// (bodies, joints, shapes, articulation links). SceneMap is the round-trip
// bridge: given an entity, where are its cooked rows; given a cooked row, which
// entity produced it. A plain hash-map wrapper -- cook fills it in, runtime and
// editing read it back. Absent fields use ~0u.
// ---------------------------------------------------------------------------

#include "scene/ecs/entity.hpp"

#include <cstdint>
#include <unordered_map>
#include <utility>

namespace nuka::scene {

struct CookedRef {
    uint32_t body_row   = ~uint32_t(0);
    uint32_t joint_row  = ~uint32_t(0);
    uint32_t dof_index  = ~uint32_t(0);
    uint32_t shape_row  = ~uint32_t(0);
    uint32_t link_index = ~uint32_t(0);
    uint32_t bp_group   = ~uint32_t(0);
};

class SceneMap {
public:
    static constexpr uint32_t kNoRow = ~uint32_t(0);

    void Bind(EntityId entity, const CookedRef& ref);

    // entity -> cooked rows (nullptr if unbound)
    const CookedRef* RefOf(EntityId entity) const;

    // cooked row -> entity (kInvalidEntity if no entity maps to that row)
    EntityId EntityOfBody(uint32_t body_row) const;
    EntityId EntityOfJoint(uint32_t joint_row) const;
    EntityId EntityOfShape(uint32_t shape_row) const;
    EntityId EntityOfLink(uint32_t link_index) const;

    // <pair> expansion: an entity pair -> the corresponding row pairs. Returns
    // kNoRow for either side when that entity is unbound / lacks the row.
    std::pair<uint32_t, uint32_t> BodyRowPair(EntityId a, EntityId b) const;
    std::pair<uint32_t, uint32_t> ShapeRowPair(EntityId a, EntityId b) const;

    void   Clear();
    size_t Size() const { return refs_.size(); }

private:
    std::unordered_map<EntityId, CookedRef, EntityIdHash> refs_;
    std::unordered_map<uint32_t, EntityId> body_to_entity_;
    std::unordered_map<uint32_t, EntityId> joint_to_entity_;
    std::unordered_map<uint32_t, EntityId> shape_to_entity_;
    std::unordered_map<uint32_t, EntityId> link_to_entity_;
};

} // namespace nuka::scene
