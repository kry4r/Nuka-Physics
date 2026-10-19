// ---------------------------------------------------------------------------
// terrain_field.cpp -- TerrainRecord -> HeightField.
// ---------------------------------------------------------------------------

#include "scene/terrain/terrain_field.hpp"

#include "scene/terrain/heightfield_loaders.hpp"

#include <stdexcept>
#include <string>

namespace nuka::terrain {

HeightField TerrainField(const scene::TerrainRecord& record) {
    HeightField hf;
    if (!record.image_path.empty()) {
        std::string err;
        if (!LoadHeightFieldFromImage(record.image_path, record.image_radius_x, record.image_radius_y,
                                      record.image_elevation_z, record.base_z, record.origin, hf, err)) {
            throw std::runtime_error("terrain image load failed: " + err);
        }
        return hf;
    }
    TerrainGenConfig cfg;
    cfg.nrow = record.nrow;            cfg.ncol = record.ncol;
    cfg.cell_x = record.cell;          cfg.cell_y = record.cell;
    cfg.origin = record.origin;        cfg.base_z = record.base_z;
    cfg.grade_x = record.grade_x;      cfg.grade_y = record.grade_y;
    cfg.ring_rise = record.ring_rise;  cfg.ring_width = record.ring_width;
    cfg.ring_platform = record.ring_platform; cfg.ring_count = record.ring_count;
    cfg.bump_height = record.bump_height;     cfg.bump_cell = record.bump_cell;
    cfg.feature_cell = record.feature_cell;   cfg.feature_margin = record.feature_margin;
    cfg.feature_seed = record.feature_seed;
    cfg.curric_levels = record.curric_levels; cfg.curric_types = record.curric_types;
    if (!GenerateHeightField(cfg, hf)) {
        throw std::runtime_error("degenerate terrain config");
    }
    return hf;
}

}  // namespace nuka::terrain
