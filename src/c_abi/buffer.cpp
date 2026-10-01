// Public state and control views expose the live model/data arenas.
// Field descriptors define dtype, element layout, and writable input storage.

#include "c_abi/dlpack_table.hpp"
#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include "nk/pipeline/world.hpp"

#include <exception>

namespace {

// Stamp the canonical per-field stride + wire dtype from dlpack_table.hpp onto
// the view. The SINGLE source of the RL binary contract's stride/dtype.
inline bool StampFieldDescriptor(nuka_state_field_t field,
                                 nuka_buffer_view_t* out) {
    const nuka::c_abi::DlpackFieldRow* row =
        nuka::c_abi::FindDlpackFieldRow(field);
    if (row == nullptr) {
        return false;
    }
    out->element_stride_bytes = row->element_stride_bytes;
    out->dtype = row->dtype;
    return true;
}

}  // namespace

extern "C" {

nuka_result_t nuka_world_get_buffer_view(nuka_world_handle world,
                                         nuka_state_field_t field,
                                         nuka_buffer_view_t* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    out->device_ptr = nullptr;
    out->element_count = 0u;
    out->element_stride_bytes = 0u;
    out->dtype = 0u;

    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }

    try {
        // Resolve the public field -> its canonical descriptor (stride/dtype +
        // the nk::FieldId bridge). An unknown public field => NOT_SUPPORTED.
        const nuka::c_abi::DlpackFieldRow* row =
            nuka::c_abi::FindDlpackFieldRow(field);
        if (row == nullptr) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }

        // A descriptor without arena storage cannot expose a device pointer.
        if (row->field_id == nuka::c_abi::kNoFieldId) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }

        // Serve the field DIRECTLY from the nk arena: the live device pointer +
        // the logical element count (env-major, stride-sized elements). Writable
        // fields (DriveTarget / the gain buffers) alias the live persistent Data
        // field -- a write IN PLACE is picked up by the NEXT Step (the RL action
        // surface contract).
        void* ptr = record->world->FieldPtr(row->field_id);
        if (ptr == nullptr) {
            if (record->world->LastStatus() != nuka::phi::Status::Ok)
                return nuka::c_abi::MapStatusToResult(record->world->LastStatus());
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        const uint64_t element_count =
            record->world->GetModel().capacities.ElementCount(row->field_id);

        out->device_ptr = ptr;
        out->element_count = element_count;
        StampFieldDescriptor(field, out);
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

// Host->field byte upload (1:1 with nk::Data::UploadField), resolving the public
// field to its nk::FieldId through the SAME dlpack_table bridge get_buffer_view uses.
nuka_result_t nuka_world_upload_field(nuka_world_handle world,
                                      nuka_state_field_t field,
                                      const void* bytes, size_t nbytes,
                                      size_t byte_offset) {
    if (bytes == nullptr && nbytes > 0u) {
        return NUKA_RESULT_INVALID_ARG;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    if (nbytes == 0u) {
        return NUKA_RESULT_OK;
    }
    try {
        const nuka::c_abi::DlpackFieldRow* row =
            nuka::c_abi::FindDlpackFieldRow(field);
        if (row == nullptr || row->field_id == nuka::c_abi::kNoFieldId) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        if (record->world->FieldPtr(row->field_id) == nullptr) {
            if (record->world->LastStatus() != nuka::phi::Status::Ok)
                return nuka::c_abi::MapStatusToResult(record->world->LastStatus());
            return NUKA_RESULT_NOT_SUPPORTED;  // capacity-0 field on this scene.
        }
        const auto status = record->world->GetData().UploadFieldStatus(
            row->field_id, bytes, static_cast<uint64_t>(nbytes),
            static_cast<uint64_t>(byte_offset));
        if (status == nuka::phi::Status::Ok &&
            row->field_id == nuka::nk::FieldId::ParticlePos) {
            const auto mirrored = record->world->GetData().UploadFieldStatus(
                nuka::nk::FieldId::ParticleKinematicTarget, bytes,
                static_cast<uint64_t>(nbytes), static_cast<uint64_t>(byte_offset));
            if (mirrored != nuka::phi::Status::Ok)
                return nuka::c_abi::MapStatusToResult(mirrored);
        }
        if (status == nuka::phi::Status::Ok &&
            (row->field_id == nuka::nk::FieldId::ParticlePos ||
             row->field_id == nuka::nk::FieldId::ParticleVel)) {
            const auto& caps = record->world->GetModel().capacities;
            if (caps.vbd_vertices_per_env > 0u) {
                const uint64_t first = byte_offset / sizeof(nuka::math::Vec3);
                const uint64_t last = (byte_offset + nbytes - 1u) / sizeof(nuka::math::Vec3);
                const uint32_t clear = 0u;
                for (uint32_t env = 0u; env < caps.env_count; ++env) {
                    const uint64_t begin = uint64_t{env} * caps.particles_per_env +
                                           caps.vbd_particle_begin;
                    const uint64_t end = begin + caps.vbd_vertices_per_env;
                    if (first >= end || last < begin) continue;
                    const auto cleared = record->world->GetData().UploadFieldStatus(
                        nuka::nk::FieldId::VbdHistoryReady, &clear, sizeof(clear),
                        uint64_t{env} * sizeof(clear));
                    if (cleared != nuka::phi::Status::Ok)
                        return nuka::c_abi::MapStatusToResult(cleared);
                }
            }
        }
        return nuka::c_abi::MapStatusToResult(status);  // over-range == LOUD.
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

// Field->host byte download (1:1 with nk::Data::DownloadField); the read mirror of
// nuka_world_upload_field with the identical resolution + LOUD over-range contract.
nuka_result_t nuka_world_download_field(nuka_world_handle world,
                                        nuka_state_field_t field,
                                        void* bytes, size_t nbytes,
                                        size_t byte_offset) {
    if (bytes == nullptr && nbytes > 0u) {
        return NUKA_RESULT_INVALID_ARG;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    if (nbytes == 0u) {
        return NUKA_RESULT_OK;
    }
    try {
        const nuka::c_abi::DlpackFieldRow* row =
            nuka::c_abi::FindDlpackFieldRow(field);
        if (row == nullptr || row->field_id == nuka::c_abi::kNoFieldId) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        if (record->world->FieldPtr(row->field_id) == nullptr) {
            if (record->world->LastStatus() != nuka::phi::Status::Ok)
                return nuka::c_abi::MapStatusToResult(record->world->LastStatus());
            return NUKA_RESULT_NOT_SUPPORTED;
        }
        const auto status = nuka::nk::LayoutOf(row->field_id).owner == nuka::nk::FieldOwner::Model
            ? record->world->GetModel().DownloadFieldStatus(row->field_id, bytes,
                static_cast<uint64_t>(nbytes), static_cast<uint64_t>(byte_offset))
            : record->world->GetData().DownloadFieldStatus(row->field_id, bytes,
                static_cast<uint64_t>(nbytes), static_cast<uint64_t>(byte_offset));
        return nuka::c_abi::MapStatusToResult(status);
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

} // extern "C"
