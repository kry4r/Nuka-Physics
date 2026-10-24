#pragma once

#include <cstdint>

#include "nk/material/field_response.hpp"

namespace nuka::nk {

enum class MaterialLaw : uint32_t { SmallStrainPlaneStressJ2Maxwell = 1u, LogStrainOrthotropicHillMaxwell = 2u };
static_assert(static_cast<uint32_t>(MaterialLaw::SmallStrainPlaneStressJ2Maxwell) == kMfSmallStrainLaw &&
              static_cast<uint32_t>(MaterialLaw::LogStrainOrthotropicHillMaxwell) == kMfLogStrainOrthotropicLaw,
              "descriptor laws must match the field laws");
enum class SectionKind : uint32_t { HomogeneousPlate = 1u, EffectiveSection = 2u };
static_assert(static_cast<uint32_t>(SectionKind::HomogeneousPlate) == kMfPlateSection &&
              static_cast<uint32_t>(SectionKind::EffectiveSection) == kMfEffectiveSection,
              "descriptor sections must match the section laws");
enum class FractureLaw : uint32_t { LegacyMidSurfaceCohesive = 1u, MixedModeMidSurfaceCohesive = 2u };
static_assert(static_cast<uint32_t>(FractureLaw::LegacyMidSurfaceCohesive) == kMfIsotropicCohesive &&
              static_cast<uint32_t>(FractureLaw::MixedModeMidSurfaceCohesive) == kMfMixedModeCohesive,
              "descriptor fracture laws must match the seam laws");
enum class HistoryKind : uint32_t { PlaneStress = 1u, Plate = 2u, Cohesive = 3u, MixedCohesive = 4u };
enum class StateRole : uint32_t { Committed, Trial, Scratch, Rebuildable };

// Material version 1 describes the small-strain law only; version 2 adds the log-strain orthotropic law.
struct ContractVersions {
    uint32_t schema = 1u;
    uint32_t material = 1u;
    uint32_t section = 1u;
    uint32_t fracture = 1u;
    uint32_t history = kMfStateVersion;
    bool Supported() const {
        return schema == 1u && (material == 1u || material == 2u) && (section == 1u || section == 2u) &&
               (fracture == 1u || fracture == 2u) && (history == 1u || history == kMfStateVersion);
    }
};

struct MaxwellBranchDescriptor {
    float stiffness_fraction = 0.0f;
    float relaxation_time = 0.0f;
};

struct MaterialDescriptor {
    MaterialLaw law = MaterialLaw::SmallStrainPlaneStressJ2Maxwell;
    uint32_t version = 1u;
    float youngs = 0.0f;
    float poisson = 0.0f;
    float yield_stress = 0.0f;
    float hardening = 0.0f;
    MaxwellBranchDescriptor maxwell[2];
    float max_surface_strain = 0.05f;
    // Version 2: cross-grain and shear moduli and Hill yield stresses in the grain frame; zero yields take defaults.
    float youngs_cd = 0.0f;
    float shear_modulus = 0.0f;
    float yield_stress_cd = 0.0f;
    float yield_shear = 0.0f;
    float yield_biaxial = 0.0f;
    bool synthetic = true;
};

// Density and reference thickness determine mass; collision and display offsets are separate data.
struct SectionDescriptor {
    SectionKind kind = SectionKind::HomogeneousPlate;
    uint32_t version = 1u;
    float reference_density = 0.0f;
    float reference_thickness = 0.0f;
    float yield_moment = 0.0f;
    float bend_hardening = 0.0f;
    MaxwellBranchDescriptor bending_maxwell[2];
    // Version 2: an effective section with its own bending stiffness (zero: the plate value) and the curvature range
    // its moment law is calibrated for.
    float bend_stiffness = 0.0f;
    float curvature_range = 0.0f;
};

// Version 1 is the isotropic mid-surface law. Version 2 separates opening along the crack normal from sliding, with
// its own sliding strength and toughness under a Benzeggagh-Kenane envelope on the sliding energy fraction.
// Neither carries the opening of a relative rotation through the thickness.
struct FractureDescriptor {
    FractureLaw law = FractureLaw::LegacyMidSurfaceCohesive;
    uint32_t version = 1u;
    float tensile_strength = 0.0f;
    float shear_strength = 0.0f;
    float reference_fracture_energy = 0.0f;
    // Version 2: sliding toughness (zero: reference_fracture_energy) and the envelope exponent (zero: 1).
    float shear_fracture_energy = 0.0f;
    float mixity_exponent = 0.0f;
};

struct MaterialCapabilities {
    bool incremental_potential = true;
    bool symmetric_consistent_tangent = true;
    bool finite_strain = false;
    bool orthotropic = false;
    bool independent_effective_bending = false;
    bool thickness_fracture_opening = false;
    bool consistent_mixed_mode_fracture = false;
};

struct StateFieldDescriptor {
    const char* name;
    const char* unit;
    uint32_t offset;
    uint32_t count;
};

inline constexpr StateFieldDescriptor kPlaneStressStateFields[] = {
    {"plastic_strain", "1", 0u, 3u}, {"equivalent_plastic_strain", "1", 3u, 1u},
    {"maxwell_strain", "1", 4u, 6u}
};
inline constexpr StateFieldDescriptor kPlateStateFields[] = {
    {"plastic_curvature", "1/m", 0u, 3u}, {"equivalent_plastic_curvature", "1/m", 3u, 1u},
    {"maxwell_curvature", "1/m", 4u, 6u}, {"hinge_damage", "1", 10u, 3u}
};
inline constexpr StateFieldDescriptor kCohesiveStateFields[] = {
    {"maximum_opening", "m", 0u, 2u}, {"inserted", "1", kMfSeamCracked, 1u}
};
inline constexpr StateFieldDescriptor kMixedCohesiveStateFields[] = {
    {"softening_knee", "1", 0u, 2u}, {"inserted", "1", kMfSeamCracked, 1u},
    {"sliding_energy_fraction", "1", kMfSeamMixity, 2u}, {"crack_normal", "1", kMfSeamNormal, 3u}
};

struct StateLayoutDescriptor {
    uint32_t version;
    HistoryKind kind;
    uint32_t scalar_count;
    const StateFieldDescriptor* fields;
    uint32_t field_count;
};

inline StateLayoutDescriptor MaterialStateLayout(HistoryKind kind) {
    switch (kind) {
        case HistoryKind::PlaneStress: return {kMfStateVersion, kind, kMfPlaneState, kPlaneStressStateFields, 3u};
        case HistoryKind::Plate: return {kMfStateVersion, kind, kMfPlateState, kPlateStateFields, 4u};
        case HistoryKind::Cohesive: return {kMfStateVersion, kind, kMfCohesiveState, kCohesiveStateFields, 2u};
        case HistoryKind::MixedCohesive:
            return {kMfStateVersion, kind, kMfMixedCohesiveState, kMixedCohesiveStateFields, 4u};
    }
    return {0u, kind, 0u, nullptr, 0u};
}

struct FieldMaterialDescriptor {
    ContractVersions versions;
    MaterialDescriptor material;
    SectionDescriptor section;
    FractureDescriptor fracture;
    MaterialCapabilities capabilities;
};

inline FieldMaterialDescriptor DescribeField(const MfMaterial& m) {
    FieldMaterialDescriptor d;
    if (m.law == kMfLogStrainOrthotropicLaw) {
        d.versions.material = 2u;
        d.material.law = MaterialLaw::LogStrainOrthotropicHillMaxwell;
        d.material.version = 2u;
        d.material.max_surface_strain = m.strain_range;
        d.material.youngs_cd = m.youngs_cd;
        d.material.shear_modulus = m.shear_modulus;
        d.material.yield_stress_cd = m.yield_stress_cd;
        d.material.yield_shear = m.yield_shear;
        d.material.yield_biaxial = m.yield_biaxial;
        d.capabilities.finite_strain = true;
        d.capabilities.orthotropic = true;
    }
    d.material.youngs = m.youngs;
    d.material.poisson = m.poisson;
    d.material.yield_stress = m.yield_stress;
    d.material.hardening = m.hardening;
    d.section.reference_density = m.density;
    d.section.reference_thickness = m.thickness;
    d.section.yield_moment = m.yield_moment;
    d.section.bend_hardening = m.bend_hardening;
    for (uint32_t i = 0; i < 2u; ++i) {
        d.material.maxwell[i] = {m.maxwell_modulus[i], m.maxwell_time[i]};
        d.section.bending_maxwell[i] = {m.bend_maxwell_modulus[i], m.bend_maxwell_time[i]};
    }
    if (m.section_law == kMfEffectiveSection) {
        d.versions.section = 2u;
        d.section.kind = SectionKind::EffectiveSection;
        d.section.version = 2u;
        d.section.bend_stiffness = m.bend_stiffness;
        d.section.curvature_range = m.curvature_range;
        d.capabilities.independent_effective_bending = true;
    }
    d.fracture.tensile_strength = m.tear_strength;
    d.fracture.shear_strength = m.shear_strength;
    d.fracture.reference_fracture_energy = m.tear_energy;
    if (m.fracture_law == kMfMixedModeCohesive) {
        d.versions.fracture = 2u;
        d.fracture.law = FractureLaw::MixedModeMidSurfaceCohesive;
        d.fracture.version = 2u;
        d.fracture.shear_fracture_energy = m.shear_tear_energy;
        d.fracture.mixity_exponent = m.mixity_exponent;
        d.capabilities.consistent_mixed_mode_fracture = true;
    }
    return d;
}

inline bool EncodeFieldMaterial(const FieldMaterialDescriptor& d, MfMaterial* out) {
    const bool small = d.material.law == MaterialLaw::SmallStrainPlaneStressJ2Maxwell;
    const bool logarithmic = d.material.law == MaterialLaw::LogStrainOrthotropicHillMaxwell;
    const bool mixed = d.fracture.law == FractureLaw::MixedModeMidSurfaceCohesive;
    const bool effective = d.section.kind == SectionKind::EffectiveSection;
    if (!out || !d.versions.Supported() || d.material.version != d.versions.material ||
        d.section.version != d.versions.section || d.fracture.version != d.versions.fracture ||
        !(small || (logarithmic && d.material.version == 2u)) ||
        !(effective ? d.section.version == 2u
                    : d.section.kind == SectionKind::HomogeneousPlate && d.section.version == 1u) ||
        !(d.fracture.law == FractureLaw::LegacyMidSurfaceCohesive ? d.fracture.version == 1u
                                                                   : mixed && d.fracture.version == 2u) ||
        (small && d.material.max_surface_strain != 0.05f))
        return false;
    MfMaterial m;
    m.law = static_cast<uint32_t>(d.material.law);
    m.fracture_law = static_cast<uint32_t>(d.fracture.law);
    if (effective) {
        m.section_law = kMfEffectiveSection;
        m.bend_stiffness = d.section.bend_stiffness;
        m.curvature_range = d.section.curvature_range;
    }
    if (mixed) {
        m.shear_tear_energy = d.fracture.shear_fracture_energy;
        m.mixity_exponent = d.fracture.mixity_exponent;
    }
    if (logarithmic) {
        m.youngs_cd = d.material.youngs_cd;
        m.shear_modulus = d.material.shear_modulus;
        m.yield_stress_cd = d.material.yield_stress_cd;
        m.yield_shear = d.material.yield_shear;
        m.yield_biaxial = d.material.yield_biaxial;
        m.strain_range = d.material.max_surface_strain;
    }
    m.youngs = d.material.youngs;
    m.poisson = d.material.poisson;
    m.yield_stress = d.material.yield_stress;
    m.hardening = d.material.hardening;
    m.density = d.section.reference_density;
    m.thickness = d.section.reference_thickness;
    m.yield_moment = d.section.yield_moment;
    m.bend_hardening = d.section.bend_hardening;
    for (uint32_t i = 0; i < 2u; ++i) {
        m.maxwell_modulus[i] = d.material.maxwell[i].stiffness_fraction;
        m.maxwell_time[i] = d.material.maxwell[i].relaxation_time;
        m.bend_maxwell_modulus[i] = d.section.bending_maxwell[i].stiffness_fraction;
        m.bend_maxwell_time[i] = d.section.bending_maxwell[i].relaxation_time;
    }
    m.tear_strength = d.fracture.tensile_strength;
    m.shear_strength = d.fracture.shear_strength;
    m.tear_energy = d.fracture.reference_fracture_energy;
    if (!mf::ValidMaterial(m)) return false;
    *out = m;
    return true;
}

}  // namespace nuka::nk
