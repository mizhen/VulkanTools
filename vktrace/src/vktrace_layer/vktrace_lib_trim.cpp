/*
* Copyright (C) 2016 Advanced Micro Devices, Inc.
*
* Author: Peter Lohrmann <Peter.Lohrmann@amd.com>
*/
#include "vktrace_lib_trim.h"
#include "vktrace_trace_packet_utils.h"
#include "vktrace_vk_vk_packets.h"
#include "vktrace_vk_packet_id.h"
#include "vk_struct_size_helper.h"
#include "vulkan.h"

// Tracks the existence of objects from the very beginning of the application
static Trim_StateTracker s_trimGlobalStateTracker;

// A snapshot of the GlobalStateTracker taken at the start of the trim frames.
static Trim_StateTracker s_trimStateTrackerSnapshot;

bool g_trimEnabled = false;
bool g_trimIsPreTrim = false;
bool g_trimIsInTrim = false;
bool g_trimIsPostTrim = false;
uint64_t g_trimFrameCounter = 0;
uint64_t g_trimStartFrame = 0;
uint64_t g_trimEndFrame = UINT64_MAX;

void trim_snapshot_state_tracker()
{
    s_trimStateTrackerSnapshot = s_trimGlobalStateTracker;
}

// List of all the packets that have been recorded for the frames of interest.
std::list<vktrace_trace_packet_header*> trim_recorded_packets;

std::unordered_map<VkCommandBuffer, std::list<vktrace_trace_packet_header*>> s_cmdBufferPackets;

#define TRIM_DEFINE_OBJECT_TRACKER_FUNCS(type) \
Trim_ObjectInfo* trim_add_##type##_object(Vk##type var) { \
   Trim_ObjectInfo& info = s_trimGlobalStateTracker.created##type##s[var]; \
   memset(&info, 0, sizeof(Trim_ObjectInfo)); \
   info.vkObject = (uint64_t)var; \
   return &info; \
} \
void trim_remove_##type##_object(Vk##type var) { \
    /* make sure the object actually existed before we attempt to remove it. This is for testing and thus only happens in debug builds. */ \
    assert(s_trimGlobalStateTracker.created##type##s.find(var) != s_trimGlobalStateTracker.created##type##s.end()); \
    s_trimGlobalStateTracker.created##type##s.erase(var); \
} \
Trim_ObjectInfo* trim_get_##type##_objectInfo(Vk##type var) { \
   TrimObjectInfoMap::iterator iter  = s_trimGlobalStateTracker.created##type##s.find(var); \
   return &(iter->second); \
}

#define TRIM_DEFINE_MARK_REF(type) \
void trim_mark_##type##_reference(Vk##type var) { \
   TrimObjectInfoMap::iterator iter  = s_trimGlobalStateTracker.created##type##s.find(var); \
   if (iter != s_trimGlobalStateTracker.created##type##s.end()) \
   { \
       iter->second.bReferencedInTrim = true; \
   } \
}


TRIM_DEFINE_MARK_REF(Instance);
TRIM_DEFINE_MARK_REF(PhysicalDevice);
TRIM_DEFINE_MARK_REF(Device);

TRIM_DEFINE_OBJECT_TRACKER_FUNCS(Instance);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(PhysicalDevice);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(Device);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(SurfaceKHR);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(Queue);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(SwapchainKHR);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(CommandPool);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(CommandBuffer);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(DeviceMemory);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(ImageView);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(Image);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(BufferView);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(Buffer);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(Sampler);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(DescriptorSetLayout);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(PipelineLayout);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(RenderPass);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(ShaderModule);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(PipelineCache);

TRIM_DEFINE_OBJECT_TRACKER_FUNCS(DescriptorPool);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(Pipeline);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(Semaphore);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(Fence);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(Framebuffer);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(Event);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(QueryPool);
TRIM_DEFINE_OBJECT_TRACKER_FUNCS(DescriptorSet);

//===============================================
// Write all stored trace packets to the trace file
//===============================================
#define TRIM_WRITE_OBJECT_PACKETS(type) \
    for (TrimObjectInfoMap::iterator iter = s_trimGlobalStateTracker.created##type##s.begin(); iter != s_trimGlobalStateTracker.created##type##s.end(); iter++) { \
        Trim_ObjectInfo info = iter->second; \
        for (std::list<vktrace_trace_packet_header*>::iterator call = info.packets.begin(); call != info.packets.end(); call++) { \
            vktrace_write_trace_packet(*call, vktrace_trace_get_trace_file()); \
        } \
    }

//===============================================
// Write trace packets only for referenced objects to the trace file
//===============================================
#define TRIM_WRITE_REFERENCED_OBJECT_PACKETS(type) \
    for (TrimObjectInfoMap::iterator iter = s_trimGlobalStateTracker.created##type##s.begin(); iter != s_trimGlobalStateTracker.created##type##s.end(); iter++) { \
        Trim_ObjectInfo info = iter->second; \
        if (info.bReferencedInTrim) { \
            for (std::list<vktrace_trace_packet_header*>::iterator call = info.packets.begin(); call != info.packets.end(); call++) { \
                if (*call != NULL) { \
                    vktrace_write_trace_packet(*call, vktrace_trace_get_trace_file()); \
                } \
            } \
            /*info.packets.clear(); */\
        } \
    } \
    /*s_trimGlobalStateTracker.created##type##s.clear(); */

//=============================================================================
// Recreate all objects
//=============================================================================
void trim_write_all_referenced_object_calls()
{
    // write the referenced objects from the snapshot
    Trim_StateTracker& stateTracker = s_trimStateTrackerSnapshot;

    // Instances (& PhysicalDevices)
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdInstances.begin(); obj != stateTracker.createdInstances.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.Instance.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Instance.pCreatePacket));

        vktrace_write_trace_packet(obj->second.ObjectInfo.Instance.pEnumeratePhysicalDevicesCountPacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Instance.pEnumeratePhysicalDevicesCountPacket));

        vktrace_write_trace_packet(obj->second.ObjectInfo.Instance.pEnumeratePhysicalDevicesPacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Instance.pEnumeratePhysicalDevicesPacket));
    }

    // SurfaceKHR
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdSurfaceKHRs.begin(); obj != stateTracker.createdSurfaceKHRs.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.SurfaceKHR.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.SurfaceKHR.pCreatePacket));
    }

    // Devices
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdDevices.begin(); obj != stateTracker.createdDevices.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.Device.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Device.pCreatePacket));
    }

    // Queue
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdQueues.begin(); obj != stateTracker.createdQueues.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.Queue.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Queue.pCreatePacket));
    }

    // CommandPool
    for (TrimObjectInfoMap::iterator poolObj = stateTracker.createdCommandPools.begin(); poolObj != stateTracker.createdCommandPools.end(); poolObj++)
    {
        vktrace_write_trace_packet(poolObj->second.ObjectInfo.CommandPool.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(poolObj->second.ObjectInfo.CommandPool.pCreatePacket));

        // Now allocate command buffers that were allocated on this pool
        for (uint32_t level = VK_COMMAND_BUFFER_LEVEL_BEGIN_RANGE; level < VK_COMMAND_BUFFER_LEVEL_END_RANGE; level++)
        {
            VkCommandBufferAllocateInfo allocateInfo;
            allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            allocateInfo.pNext = NULL;
            allocateInfo.commandPool = (VkCommandPool)poolObj->first;
            allocateInfo.level = (VkCommandBufferLevel)level;
            allocateInfo.commandBufferCount = poolObj->second.ObjectInfo.CommandPool.numCommandBuffersAllocated[level];
            if (allocateInfo.commandBufferCount > 0)
            {
                VkCommandBuffer* pCommandBuffers = new VkCommandBuffer[allocateInfo.commandBufferCount];
                uint32_t index = 0;
                for (TrimObjectInfoMap::iterator cbIter = stateTracker.createdCommandBuffers.begin(); cbIter != stateTracker.createdCommandBuffers.end(); cbIter++)
                {
                    if (cbIter->second.ObjectInfo.CommandBuffer.commandPool == (VkCommandPool)poolObj->first &&
                        cbIter->second.ObjectInfo.CommandBuffer.level == level)
                    {
                        pCommandBuffers[index] = (VkCommandBuffer)cbIter->first;
                        index++;
                    }
                }

                vktrace_trace_packet_header* pHeader;
                packet_vkAllocateCommandBuffers* pPacket = NULL;
                CREATE_TRACE_PACKET(vkAllocateCommandBuffers, get_struct_chain_size(&allocateInfo) + sizeof(VkCommandBuffer) * allocateInfo.commandBufferCount);
                vktrace_set_packet_entrypoint_end_time(pHeader);
                pPacket = interpret_body_as_vkAllocateCommandBuffers(pHeader);
                pPacket->device = poolObj->second.belongsToDevice;
                vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocateInfo), sizeof(VkCommandBufferAllocateInfo), &allocateInfo);
                vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pCommandBuffers), sizeof(VkCommandBuffer) * allocateInfo.commandBufferCount, pCommandBuffers);
                pPacket->result = VK_SUCCESS;
                vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocateInfo));
                vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pCommandBuffers));
                FINISH_TRACE_PACKET();

                delete[] pCommandBuffers;
            }
        }
    }

    // SwapchainKHR
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdSwapchainKHRs.begin(); obj != stateTracker.createdSwapchainKHRs.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.SwapchainKHR.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.SwapchainKHR.pCreatePacket));

        vktrace_write_trace_packet(obj->second.ObjectInfo.SwapchainKHR.pGetSwapchainImageCountPacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.SwapchainKHR.pGetSwapchainImageCountPacket));

        vktrace_write_trace_packet(obj->second.ObjectInfo.SwapchainKHR.pGetSwapchainImagesPacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.SwapchainKHR.pGetSwapchainImagesPacket));
    }

    // DeviceMemory
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdDeviceMemorys.begin(); obj != stateTracker.createdDeviceMemorys.end(); obj++)
    {
        // AllocateMemory
        vktrace_write_trace_packet(obj->second.ObjectInfo.DeviceMemory.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.DeviceMemory.pCreatePacket));

        // will need to map / unmap and set the memory contents
    }

    // Image
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdImages.begin(); obj != stateTracker.createdImages.end(); obj++)
    {
        // CreateImage
        if (obj->second.ObjectInfo.Image.pCreatePacket != NULL)
        {
            vktrace_write_trace_packet(obj->second.ObjectInfo.Image.pCreatePacket, vktrace_trace_get_trace_file());
            vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Image.pCreatePacket));
        }

        // BindImageMemory
        if (obj->second.ObjectInfo.Image.pBindImageMemoryPacket != NULL)
        {
            vktrace_write_trace_packet(obj->second.ObjectInfo.Image.pBindImageMemoryPacket, vktrace_trace_get_trace_file());
            vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Image.pBindImageMemoryPacket));
        }
    }

    // ImageView
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdImageViews.begin(); obj != stateTracker.createdImageViews.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.ImageView.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.ImageView.pCreatePacket));
    }

    // Buffer
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdBuffers.begin(); obj != stateTracker.createdBuffers.end(); obj++)
    {
        // CreateBuffer
        if (obj->second.ObjectInfo.Buffer.pCreatePacket != NULL)
        {
            vktrace_write_trace_packet(obj->second.ObjectInfo.Buffer.pCreatePacket, vktrace_trace_get_trace_file());
            vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Buffer.pCreatePacket));
        }

        // BindBufferMemory
        if (obj->second.ObjectInfo.Buffer.pBindBufferMemoryPacket != NULL)
        {
            vktrace_write_trace_packet(obj->second.ObjectInfo.Buffer.pBindBufferMemoryPacket, vktrace_trace_get_trace_file());
            vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Buffer.pBindBufferMemoryPacket));
        }
    }

    // BufferView
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdBufferViews.begin(); obj != stateTracker.createdBufferViews.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.BufferView.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.BufferView.pCreatePacket));
    }

    // Sampler
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdSamplers.begin(); obj != stateTracker.createdSamplers.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.Sampler.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Sampler.pCreatePacket));
    }

    // DescriptorSetLayout
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdDescriptorSetLayouts.begin(); obj != stateTracker.createdDescriptorSetLayouts.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.DescriptorSetLayout.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.DescriptorSetLayout.pCreatePacket));
    }

    // PipelineLayout
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdPipelineLayouts.begin(); obj != stateTracker.createdPipelineLayouts.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.PipelineLayout.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.PipelineLayout.pCreatePacket));
    }

    // RenderPass
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdRenderPasss.begin(); obj != stateTracker.createdRenderPasss.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.RenderPass.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.RenderPass.pCreatePacket));
    }

    // ShaderModule
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdShaderModules.begin(); obj != stateTracker.createdShaderModules.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.ShaderModule.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.ShaderModule.pCreatePacket));
    }

    // PipelineCache
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdPipelineCaches.begin(); obj != stateTracker.createdPipelineCaches.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.PipelineCache.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.PipelineCache.pCreatePacket));
    }

    // Pipeline
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdPipelines.begin(); obj != stateTracker.createdPipelines.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.Pipeline.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Pipeline.pCreatePacket));
    }

    // DescriptorPool
    for (TrimObjectInfoMap::iterator poolObj = stateTracker.createdDescriptorPools.begin(); poolObj != stateTracker.createdDescriptorPools.end(); poolObj++)
    {
        // write the createDescriptorPool packet
        vktrace_write_trace_packet(poolObj->second.ObjectInfo.DescriptorPool.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(poolObj->second.ObjectInfo.DescriptorPool.pCreatePacket));

        // now allocate all DescriptorSets that are part of this pool
        vktrace_trace_packet_header* pHeader;
        packet_vkAllocateDescriptorSets* pPacket = NULL;
        uint64_t vktraceStartTime = vktrace_get_time();
        SEND_ENTRYPOINT_ID(vkAllocateDescriptorSets);
        VkDescriptorSetAllocateInfo allocateInfo;
        allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocateInfo.pNext = NULL;
        allocateInfo.descriptorPool = (VkDescriptorPool)poolObj->first;
        allocateInfo.descriptorSetCount = poolObj->second.ObjectInfo.DescriptorPool.numSets;

        VkDescriptorSetLayout* pSetLayouts = new VkDescriptorSetLayout[allocateInfo.descriptorSetCount];
        allocateInfo.pSetLayouts = pSetLayouts;
        VkDescriptorSet* pDescriptorSets = new VkDescriptorSet[allocateInfo.descriptorSetCount];

        uint32_t index = 0;

        for (TrimObjectInfoMap::iterator setObj = stateTracker.createdDescriptorSets.begin(); setObj != stateTracker.createdDescriptorSets.end(); setObj++)
        {
            if (setObj->second.ObjectInfo.DescriptorSet.descriptorPool == allocateInfo.descriptorPool)
            {
                pSetLayouts[index] = setObj->second.ObjectInfo.DescriptorSet.layout;
                pDescriptorSets[index] = (VkDescriptorSet)setObj->first;
                index++;
            }
        }

        CREATE_TRACE_PACKET(vkAllocateDescriptorSets, vk_size_vkdescriptorsetallocateinfo(&allocateInfo) + (allocateInfo.descriptorSetCount * sizeof(VkDescriptorSet)));
        pHeader->vktrace_begin_time = vktraceStartTime;

        pHeader->entrypoint_begin_time = vktrace_get_time();
        pHeader->entrypoint_end_time = vktrace_get_time();
        pPacket = interpret_body_as_vkAllocateDescriptorSets(pHeader);
        pPacket->device = poolObj->second.belongsToDevice;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocateInfo), sizeof(VkDescriptorSetAllocateInfo), &allocateInfo);
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocateInfo->pSetLayouts), allocateInfo.descriptorSetCount * sizeof(VkDescriptorSetLayout), allocateInfo.pSetLayouts);
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pDescriptorSets), allocateInfo.descriptorSetCount * sizeof(VkDescriptorSet), pDescriptorSets);
        pPacket->result = VK_SUCCESS;
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocateInfo->pSetLayouts));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pDescriptorSets));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocateInfo));
        FINISH_TRACE_PACKET();

        delete[] pSetLayouts;
        delete[] pDescriptorSets;
    }

    // Framebuffer
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdFramebuffers.begin(); obj != stateTracker.createdFramebuffers.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.Framebuffer.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Framebuffer.pCreatePacket));
    }

    // Semaphore
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdSemaphores.begin(); obj != stateTracker.createdSemaphores.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.Semaphore.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Semaphore.pCreatePacket));
    }

    // Fence
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdFences.begin(); obj != stateTracker.createdFences.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.Fence.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Fence.pCreatePacket));
    }

    // Event
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdEvents.begin(); obj != stateTracker.createdEvents.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.Event.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.Event.pCreatePacket));
    }

    // QueryPool
    for (TrimObjectInfoMap::iterator obj = stateTracker.createdQueryPools.begin(); obj != stateTracker.createdQueryPools.end(); obj++)
    {
        vktrace_write_trace_packet(obj->second.ObjectInfo.QueryPool.pCreatePacket, vktrace_trace_get_trace_file());
        vktrace_delete_trace_packet(&(obj->second.ObjectInfo.QueryPool.pCreatePacket));
    }

    // write out the packets to recreate the command buffers that were just allocated
    for (TrimObjectInfoMap::iterator cmdBuffer = stateTracker.createdCommandBuffers.begin(); cmdBuffer != stateTracker.createdCommandBuffers.end(); cmdBuffer++)
    {
        std::list<vktrace_trace_packet_header*>& packets = s_cmdBufferPackets[(VkCommandBuffer)cmdBuffer->first];
        for (std::list<vktrace_trace_packet_header*>::iterator packet = packets.begin(); packet != packets.end(); packet++)
        {
            vktrace_trace_packet_header* pHeader = *packet;
            vktrace_write_trace_packet(pHeader, vktrace_trace_get_trace_file());
            vktrace_delete_trace_packet(&pHeader);
        }
    }
}

#define TRIM_ADD_OBJECT_CALL(type) \
void trim_add_##type##_call(Vk##type var, vktrace_trace_packet_header* pHeader) { \
    /* if it's in the created list, add it there*/ \
    TrimObjectInfoMap::iterator iter = s_trimGlobalStateTracker.created##type##s.find(var); \
    /*assert(iter != s_trimGlobalStateTracker.created##type##s.end()); */ \
    /*if (iter != s_trimGlobalStateTracker.created##type##s.end() ) { iter->second.packets.push_back(pHeader); } */ \
}

#define TRIM_MARK_OBJECT_REFERENCE(type) \
void trim_mark_##type##_reference(Vk##type var) { \
    Trim_ObjectInfo* info = &s_trimGlobalStateTracker.created##type##s[var]; \
    info->bReferencedInTrim = true; \
}

#define TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(type) \
void trim_mark_##type##_reference(Vk##type var) { \
    Trim_ObjectInfo* info = &s_trimGlobalStateTracker.created##type##s[var]; \
    info->bReferencedInTrim = true; \
    trim_mark_Device_reference((VkDevice)info->belongsToDevice); \
}

//===============================================
// Object tracking
//===============================================
void trim_add_Instance_call(VkInstance var, vktrace_trace_packet_header* pHeader) {
//    TrimObjectInfoMap::iterator iter = s_trimGlobalStateTracker.createdInstances.find(var);
//    assert(iter != s_trimGlobalStateTracker.createdInstances.end());
    //if (iter != s_trimGlobalStateTracker.createdInstances.end())
    //{
    //    iter->second.packets.push_back(pHeader);
    //}
}

TRIM_ADD_OBJECT_CALL(PhysicalDevice)
TRIM_ADD_OBJECT_CALL(Device)

TRIM_ADD_OBJECT_CALL(CommandPool)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(CommandPool)

//TRIM_ADD_OBJECT_CALL(CommandBuffer)
void trim_add_CommandBuffer_call(VkCommandBuffer var, vktrace_trace_packet_header* pHeader) {
    s_cmdBufferPackets[var].push_back(pHeader);
}

void trim_remove_CommandBuffer_calls(VkCommandBuffer var)
{
    s_cmdBufferPackets.erase(var);
}

TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(CommandBuffer)

TRIM_ADD_OBJECT_CALL(DescriptorPool)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(DescriptorPool)

TRIM_ADD_OBJECT_CALL(DescriptorSet)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(DescriptorSet)

TRIM_ADD_OBJECT_CALL(RenderPass)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(RenderPass)

TRIM_ADD_OBJECT_CALL(PipelineCache)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(PipelineCache)

TRIM_ADD_OBJECT_CALL(Pipeline)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(Pipeline)

TRIM_ADD_OBJECT_CALL(Queue)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(Queue);

TRIM_ADD_OBJECT_CALL(Semaphore)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(Semaphore)

TRIM_ADD_OBJECT_CALL(DeviceMemory)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(DeviceMemory)

TRIM_ADD_OBJECT_CALL(Fence)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(Fence)

TRIM_ADD_OBJECT_CALL(SwapchainKHR)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(SwapchainKHR)

TRIM_ADD_OBJECT_CALL(Image)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(Image)

TRIM_ADD_OBJECT_CALL(ImageView)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(ImageView)

TRIM_ADD_OBJECT_CALL(Buffer)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(Buffer)

TRIM_ADD_OBJECT_CALL(BufferView)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(BufferView)

TRIM_ADD_OBJECT_CALL(Framebuffer)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(Framebuffer)

TRIM_ADD_OBJECT_CALL(Event)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(Event)

TRIM_ADD_OBJECT_CALL(QueryPool)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(QueryPool)

TRIM_ADD_OBJECT_CALL(ShaderModule)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(ShaderModule)

TRIM_ADD_OBJECT_CALL(PipelineLayout)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(PipelineLayout)

TRIM_ADD_OBJECT_CALL(Sampler)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(Sampler)

TRIM_ADD_OBJECT_CALL(DescriptorSetLayout)
TRIM_MARK_OBJECT_REFERENCE_WITH_DEVICE_DEPENDENCY(DescriptorSetLayout)

//===============================================
// Packet Recording for frames of interest
//===============================================
void trim_add_recorded_packet(vktrace_trace_packet_header* pHeader)
{
    trim_recorded_packets.push_back(pHeader);
}

void trim_write_recorded_packets()
{
    for (std::list<vktrace_trace_packet_header*>::iterator call = trim_recorded_packets.begin(); call != trim_recorded_packets.end(); call++)
    {
        vktrace_write_trace_packet(*call, vktrace_trace_get_trace_file());
    }
}


//===============================================
// Write packets to destroy all created created objects.
// Remember that we want to destroy them roughly in the opposite 
// order they were created, so that means the Instance is the last
// object to destroy!
//===============================================
void trim_write_destroy_packets()
{
    // QueryPool
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdQueryPools.begin(); obj != s_trimGlobalStateTracker.createdQueryPools.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyQueryPool* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyQueryPool, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyQueryPool(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->queryPool = (VkQueryPool)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.QueryPool.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // Event
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdEvents.begin(); obj != s_trimGlobalStateTracker.createdEvents.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyEvent* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyEvent, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyEvent(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->event = (VkEvent)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.Event.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // Fence
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdFences.begin(); obj != s_trimGlobalStateTracker.createdFences.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyFence* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyFence, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyFence(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->fence = (VkFence)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.Fence.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // Semaphore
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdSemaphores.begin(); obj != s_trimGlobalStateTracker.createdSemaphores.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroySemaphore* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroySemaphore, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroySemaphore(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->semaphore = (VkSemaphore)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.Semaphore.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // Framebuffer
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdFramebuffers.begin(); obj != s_trimGlobalStateTracker.createdFramebuffers.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyFramebuffer* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyFramebuffer, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyFramebuffer(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->framebuffer = (VkFramebuffer)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.Framebuffer.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // DescriptorPool
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdDescriptorPools.begin(); obj != s_trimGlobalStateTracker.createdDescriptorPools.end(); obj++)
    {
        // Free the associated DescriptorSets
        VkDescriptorPool descriptorPool = (VkDescriptorPool)obj->first;
        uint32_t descriptorSetCount = obj->second.ObjectInfo.DescriptorPool.numSets;
        if (descriptorSetCount > 0)
        {
            vktrace_trace_packet_header* pHeader;
            packet_vkFreeDescriptorSets* pPacket = NULL;
            CREATE_TRACE_PACKET(vkFreeDescriptorSets, descriptorSetCount*sizeof(VkDescriptorSet));
            vktrace_set_packet_entrypoint_end_time(pHeader);
            pPacket = interpret_body_as_vkFreeDescriptorSets(pHeader);
            pPacket->device = obj->second.belongsToDevice;
            pPacket->descriptorPool = descriptorPool;
            pPacket->descriptorSetCount = descriptorSetCount;

            VkDescriptorSet* pDescriptorSets = new VkDescriptorSet[descriptorSetCount];
            uint32_t index = 0;
            for (TrimObjectInfoMap::iterator dsIter = s_trimGlobalStateTracker.createdDescriptorSets.begin(); dsIter != s_trimGlobalStateTracker.createdDescriptorSets.end(); dsIter++)
            {
                if (dsIter->second.ObjectInfo.DescriptorSet.descriptorPool == (VkDescriptorPool)obj->first)
                {
                    pDescriptorSets[index] = (VkDescriptorSet)dsIter->first;
                    index++;
                }
            }

            vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pDescriptorSets), descriptorSetCount*sizeof(VkDescriptorSet), pDescriptorSets);
            pPacket->result = VK_SUCCESS;
            vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pDescriptorSets));
            FINISH_TRACE_PACKET();

            delete[] pDescriptorSets;
        }

        // Now destroy the DescriptorPool
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyDescriptorPool* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyDescriptorPool, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyDescriptorPool(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->descriptorPool = (VkDescriptorPool)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.DescriptorPool.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // Pipeline
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdPipelines.begin(); obj != s_trimGlobalStateTracker.createdPipelines.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyPipeline* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyPipeline, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyPipeline(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->pipeline = (VkPipeline)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.Pipeline.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // PipelineCache
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdPipelineCaches.begin(); obj != s_trimGlobalStateTracker.createdPipelineCaches.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyPipelineCache* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyPipelineCache, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyPipelineCache(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->pipelineCache = (VkPipelineCache)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.PipelineCache.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // ShaderModule
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdShaderModules.begin(); obj != s_trimGlobalStateTracker.createdShaderModules.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyShaderModule* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyShaderModule, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyShaderModule(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->shaderModule = (VkShaderModule)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.ShaderModule.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // RenderPass
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdRenderPasss.begin(); obj != s_trimGlobalStateTracker.createdRenderPasss.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyRenderPass* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyRenderPass, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyRenderPass(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->renderPass = (VkRenderPass)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.RenderPass.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // PipelineLayout
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdPipelineLayouts.begin(); obj != s_trimGlobalStateTracker.createdPipelineLayouts.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyPipelineLayout* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyPipelineLayout, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyPipelineLayout(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->pipelineLayout = (VkPipelineLayout)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.PipelineLayout.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // DescriptorSetLayout
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdSamplers.begin(); obj != s_trimGlobalStateTracker.createdSamplers.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyDescriptorSetLayout* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyDescriptorSetLayout, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyDescriptorSetLayout(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->descriptorSetLayout = (VkDescriptorSetLayout)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.DescriptorSetLayout.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // Sampler
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdSamplers.begin(); obj != s_trimGlobalStateTracker.createdSamplers.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroySampler* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroySampler, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroySampler(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->sampler = (VkSampler)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.Sampler.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // Buffer
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdBuffers.begin(); obj != s_trimGlobalStateTracker.createdBuffers.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyBuffer* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyBuffer, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyBuffer(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->buffer = (VkBuffer)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.Buffer.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // BufferView
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdBufferViews.begin(); obj != s_trimGlobalStateTracker.createdBufferViews.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyBufferView* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyBufferView, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyBufferView(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->bufferView = (VkBufferView)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.BufferView.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // Image
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdImages.begin(); obj != s_trimGlobalStateTracker.createdImages.end(); obj++)
    {
        if (obj->second.ObjectInfo.Image.bIsSwapchainImage == false)
        {
            vktrace_trace_packet_header* pHeader;
            packet_vkDestroyImage* pPacket = NULL;
            CREATE_TRACE_PACKET(vkDestroyImage, sizeof(VkAllocationCallbacks));
            vktrace_set_packet_entrypoint_end_time(pHeader);
            pPacket = interpret_body_as_vkDestroyImage(pHeader);
            pPacket->device = obj->second.belongsToDevice;
            pPacket->image = (VkImage)obj->first;
            vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.Image.allocator));
            vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
            FINISH_TRACE_PACKET();
        }
    }

    // ImageView
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdImageViews.begin(); obj != s_trimGlobalStateTracker.createdImageViews.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyImageView* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyImageView, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyImageView(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->imageView = (VkImageView)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.ImageView.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // DeviceMemory
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdDeviceMemorys.begin(); obj != s_trimGlobalStateTracker.createdDeviceMemorys.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkFreeMemory* pPacket = NULL;
        CREATE_TRACE_PACKET(vkFreeMemory, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkFreeMemory(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->memory = (VkDeviceMemory)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.DeviceMemory.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // SwapchainKHR
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdSwapchainKHRs.begin(); obj != s_trimGlobalStateTracker.createdSwapchainKHRs.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroySwapchainKHR* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroySwapchainKHR, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroySwapchainKHR(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->swapchain = (VkSwapchainKHR)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.SwapchainKHR.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // CommandPool
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdCommandPools.begin(); obj != s_trimGlobalStateTracker.createdCommandPools.end(); obj++)
    {
        // free the command buffers
        for (uint32_t level = VK_COMMAND_BUFFER_LEVEL_BEGIN_RANGE; level < VK_COMMAND_BUFFER_LEVEL_END_RANGE; level++)
        {
            uint32_t commandBufferCount = obj->second.ObjectInfo.CommandPool.numCommandBuffersAllocated[level];
            if (commandBufferCount > 0)
            {
                vktrace_trace_packet_header* pHeader;
                packet_vkFreeCommandBuffers* pPacket = NULL;
                CREATE_TRACE_PACKET(vkFreeCommandBuffers, commandBufferCount*sizeof(VkCommandBuffer));
                vktrace_set_packet_entrypoint_end_time(pHeader);
                pPacket = interpret_body_as_vkFreeCommandBuffers(pHeader);
                pPacket->device = obj->second.belongsToDevice;
                pPacket->commandPool = (VkCommandPool)obj->first;
                pPacket->commandBufferCount = commandBufferCount;

                VkCommandBuffer* pCommandBuffers = new VkCommandBuffer[commandBufferCount];
                uint32_t index = 0;
                for (TrimObjectInfoMap::iterator cbIter = s_trimGlobalStateTracker.createdCommandBuffers.begin(); cbIter != s_trimGlobalStateTracker.createdCommandBuffers.end(); cbIter++)
                {
                    if (cbIter->second.ObjectInfo.CommandBuffer.commandPool == (VkCommandPool)obj->first &&
                        cbIter->second.ObjectInfo.CommandBuffer.level == level)
                    {
                        pCommandBuffers[index] = (VkCommandBuffer)cbIter->first;
                        index++;
                    }
                }

                vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pCommandBuffers), commandBufferCount*sizeof(VkCommandBuffer), pCommandBuffers);
                vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pCommandBuffers));
                FINISH_TRACE_PACKET();

                delete[] pCommandBuffers;
            }
        }

        // Destroy the commandPool
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyCommandPool* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyCommandPool, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyCommandPool(pHeader);
        pPacket->device = obj->second.belongsToDevice;
        pPacket->commandPool = (VkCommandPool) obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.CommandPool.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // Devices
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdDevices.begin(); obj != s_trimGlobalStateTracker.createdDevices.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyDevice* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyDevice, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyDevice(pHeader);
        pPacket->device = (VkDevice)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.Device.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // SurfaceKHR
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdSurfaceKHRs.begin(); obj != s_trimGlobalStateTracker.createdSurfaceKHRs.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroySurfaceKHR* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroySurfaceKHR, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroySurfaceKHR(pHeader);
        pPacket->surface = (VkSurfaceKHR)obj->first;
        pPacket->instance = obj->second.belongsToInstance;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.SurfaceKHR.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }

    // Instances
    for (TrimObjectInfoMap::iterator obj = s_trimGlobalStateTracker.createdInstances.begin(); obj != s_trimGlobalStateTracker.createdInstances.end(); obj++)
    {
        vktrace_trace_packet_header* pHeader;
        packet_vkDestroyInstance* pPacket = NULL;
        CREATE_TRACE_PACKET(vkDestroyInstance, sizeof(VkAllocationCallbacks));
        vktrace_set_packet_entrypoint_end_time(pHeader);
        pPacket = interpret_body_as_vkDestroyInstance(pHeader);
        pPacket->instance = (VkInstance)obj->first;
        vktrace_add_buffer_to_trace_packet(pHeader, (void**)&(pPacket->pAllocator), sizeof(VkAllocationCallbacks), &(obj->second.ObjectInfo.Instance.allocator));
        vktrace_finalize_buffer_address(pHeader, (void**)&(pPacket->pAllocator));
        FINISH_TRACE_PACKET();
    }
}

//===============================================
// Delete all the created packets
//===============================================
#define TRIM_DELETE_ALL_PACKETS(type) \
    for (TrimObjectInfoMap::iterator iter = s_trimGlobalStateTracker.created##type##s.begin(); iter != s_trimGlobalStateTracker.created##type##s.end(); iter++) { \
/*        for (std::list<vktrace_trace_packet_header*>::iterator call = iter->second.packets.begin(); call != iter->second.packets.end(); call++) { \
            if (*call != NULL) { \
                vktrace_delete_trace_packet(&(*call)); \
            } \
        } \
        iter->second.packets.clear(); */\
    } \
    s_trimGlobalStateTracker.created##type##s.clear();

void trim_delete_all_packets()
{
    // delete all recorded packets
    for (std::list<vktrace_trace_packet_header*>::iterator call = trim_recorded_packets.begin(); call != trim_recorded_packets.end(); call++)
    {
        vktrace_delete_trace_packet(&(*call));
    }
    trim_recorded_packets.clear();


    TRIM_DELETE_ALL_PACKETS(Instance);
    TRIM_DELETE_ALL_PACKETS(PhysicalDevice);
    TRIM_DELETE_ALL_PACKETS(Device);
    TRIM_DELETE_ALL_PACKETS(CommandPool);
    TRIM_DELETE_ALL_PACKETS(SwapchainKHR);
    TRIM_DELETE_ALL_PACKETS(Queue);
    TRIM_DELETE_ALL_PACKETS(DeviceMemory);
    TRIM_DELETE_ALL_PACKETS(Image);
    TRIM_DELETE_ALL_PACKETS(ImageView);
    TRIM_DELETE_ALL_PACKETS(Buffer);
    TRIM_DELETE_ALL_PACKETS(BufferView);
    TRIM_DELETE_ALL_PACKETS(ShaderModule);
    TRIM_DELETE_ALL_PACKETS(Sampler);
    TRIM_DELETE_ALL_PACKETS(RenderPass);
    TRIM_DELETE_ALL_PACKETS(Framebuffer);
    TRIM_DELETE_ALL_PACKETS(DescriptorSetLayout);
    TRIM_DELETE_ALL_PACKETS(DescriptorPool);
    TRIM_DELETE_ALL_PACKETS(DescriptorSet);
    TRIM_DELETE_ALL_PACKETS(PipelineLayout);
    TRIM_DELETE_ALL_PACKETS(Pipeline);
    TRIM_DELETE_ALL_PACKETS(Semaphore);
    TRIM_DELETE_ALL_PACKETS(Fence);
    TRIM_DELETE_ALL_PACKETS(Event);
    TRIM_DELETE_ALL_PACKETS(QueryPool);
    TRIM_DELETE_ALL_PACKETS(CommandBuffer);
}