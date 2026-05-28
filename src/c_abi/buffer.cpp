#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include <exception>

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

    try {
        const nuka_result_t refresh_result = nuka::c_abi::RefreshWorldBuffers(*record);
        if (refresh_result != NUKA_RESULT_OK) {
            return refresh_result;
        }

        if (field == NUKA_FIELD_JOINT_POSITION) {
            out->device_ptr = record->joint_position_buffer.Data();
            out->element_count = record->joint_position_buffer.Size() / sizeof(float);
            out->element_stride_bytes = sizeof(float);
            out->dtype = 0u;
            return NUKA_RESULT_OK;
        }
        if (field == NUKA_FIELD_JOINT_VELOCITY) {
            out->device_ptr = record->joint_velocity_buffer.Data();
            out->element_count = record->joint_velocity_buffer.Size() / sizeof(float);
            out->element_stride_bytes = sizeof(float);
            out->dtype = 0u;
            return NUKA_RESULT_OK;
        }
        return NUKA_RESULT_NOT_SUPPORTED;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

} // extern "C"
