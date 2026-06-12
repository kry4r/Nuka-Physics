// ---------------------------------------------------------------------------
// nuka::scene::SceneMap implementation.
// ---------------------------------------------------------------------------

#include "scene/scene_map.hpp"

namespace nuka::scene {

void SceneMap::Bind(EntityId entity, const CookedRef& ref) {
    refs_[entity] = ref;
    if (ref.body_row   != kNoRow) body_to_entity_[ref.body_row]   = entity;
    if (ref.joint_row  != kNoRow) joint_to_entity_[ref.joint_row] = entity;
    if (ref.shape_row  != kNoRow) shape_to_entity_[ref.shape_row] = entity;
    if (ref.link_index != kNoRow) link_to_entity_[ref.link_index] = entity;
}

const CookedRef* SceneMap::RefOf(EntityId entity) const {
    const auto it = refs_.find(entity);
    return it != refs_.end() ? &it->second : nullptr;
}

EntityId SceneMap::EntityOfBody(uint32_t body_row) const {
    const auto it = body_to_entity_.find(body_row);
    return it != body_to_entity_.end() ? it->second : kInvalidEntity;
}

EntityId SceneMap::EntityOfJoint(uint32_t joint_row) const {
    const auto it = joint_to_entity_.find(joint_row);
    return it != joint_to_entity_.end() ? it->second : kInvalidEntity;
}

EntityId SceneMap::EntityOfShape(uint32_t shape_row) const {
    const auto it = shape_to_entity_.find(shape_row);
    return it != shape_to_entity_.end() ? it->second : kInvalidEntity;
}

EntityId SceneMap::EntityOfLink(uint32_t link_index) const {
    const auto it = link_to_entity_.find(link_index);
    return it != link_to_entity_.end() ? it->second : kInvalidEntity;
}

std::pair<uint32_t, uint32_t> SceneMap::BodyRowPair(EntityId a, EntityId b) const {
    const auto* ra = RefOf(a);
    const auto* rb = RefOf(b);
    return {ra ? ra->body_row : kNoRow, rb ? rb->body_row : kNoRow};
}

std::pair<uint32_t, uint32_t> SceneMap::ShapeRowPair(EntityId a, EntityId b) const {
    const auto* ra = RefOf(a);
    const auto* rb = RefOf(b);
    return {ra ? ra->shape_row : kNoRow, rb ? rb->shape_row : kNoRow};
}

void SceneMap::Clear() {
    refs_.clear();
    body_to_entity_.clear();
    joint_to_entity_.clear();
    shape_to_entity_.clear();
    link_to_entity_.clear();
}

} // namespace nuka::scene
