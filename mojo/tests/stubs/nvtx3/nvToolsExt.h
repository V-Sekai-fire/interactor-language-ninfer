#pragma once
// Test stub: NVTX is a profiling API from the CUDA toolkit; the frontend headers only pull it in.
#include <cstdint>
typedef void* nvtxDomainHandle_t;
typedef void* nvtxStringHandle_t;
typedef std::uint64_t nvtxRangeId_t;
#define NVTX_VERSION 3
#define NVTX_EVENT_ATTRIB_STRUCT_SIZE 0
#define NVTX_COLOR_ARGB 1
#define NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 1
#define NVTX_MESSAGE_TYPE_REGISTERED 3
struct nvtxEventAttributes_t {
    std::uint16_t version, size; std::uint32_t category; int colorType; std::uint32_t color; int payloadType;
    union { std::uint64_t ullValue; } payload; int messageType; union { nvtxStringHandle_t registered; } message;
};
inline nvtxDomainHandle_t nvtxDomainCreateA(const char*) { return nullptr; }
inline void nvtxDomainNameCategoryA(nvtxDomainHandle_t, std::uint32_t, const char*) {}
inline nvtxStringHandle_t nvtxDomainRegisterStringA(nvtxDomainHandle_t, const char*) { return nullptr; }
inline int nvtxDomainRangePushEx(nvtxDomainHandle_t, const nvtxEventAttributes_t*) { return 0; }
inline int nvtxDomainRangePop(nvtxDomainHandle_t) { return 0; }
inline nvtxRangeId_t nvtxDomainRangeStartEx(nvtxDomainHandle_t, const nvtxEventAttributes_t*) { return 0; }
inline void nvtxDomainRangeEnd(nvtxDomainHandle_t, nvtxRangeId_t) {}
