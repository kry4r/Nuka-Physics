// ---------------------------------------------------------------------------
// nuka::c_abi -- the RL zero-copy state-buffer view (M9 T5/T6 cutover).
//
// Every public state field is now served DIRECTLY from the ONE generic
// nk::World's Data/Model arena: device_ptr = world->FieldPtr(field_id),
// element_count = Model::capacities.ElementCount(field_id) (the logical
// stride-sized element count, env-major). The per-field stride/dtype/FieldId come
// from the single source dlpack_table.hpp (the RL binary contract) -- BYTE-
// IDENTICAL to the legacy path. The legacy single-env host round-trip and the
// batched-world device-buffer arms are deleted.
//
// T1 (unified actuator): TORQUE_INPUT now ALIASES nk::FieldId::DriveTarget in
// dlpack_table.hpp -- the Torque preset reads its `u` from the SAME persistent
// Data field as the PD target, so TORQUE_INPUT is served zero-copy as the very
// same device buffer as DRIVE_TARGET (one control buffer, preset-reinterpreted).
// The remaining three control-input fields (VELOCITY_TARGET/ACTUATOR_NOLOAD_SPEED/
// TASK_TARGET) still map to kNoFieldId -- their presets are not yet wired onto the
// unified nk::World, so they return NOT_SUPPORTED rather than aliasing a buffer
// that no preset consumes yet.
// ---------------------------------------------------------------------------

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

        // The control-input params buffers (TORQUE_INPUT/VELOCITY_TARGET/
        // ACTUATOR_NOLOAD_SPEED/TASK_TARGET) have no Arena field (kNoFieldId);
        // their source was the deleted batched world. M10 named gap (non-PD
        // control). Honest NOT_SUPPORTED rather than a dangling alias.
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
            // Field has no allocated storage in this Model (e.g. a contact/SDF
            // field on a scene with that capacity == 0). Honest NOT_SUPPORTED.
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

} // extern "C"
