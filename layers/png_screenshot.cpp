/*
 *
 * Copyright (C) 2016 Advanced Micro Devices, Inc.
 * Copyright (C) 2015-2016 Valve Corporation
 * Copyright (C) 2015-2016 LunarG, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 * Auther: Peter Lohrmann <peter.lohrmann@amd.com>
 *
 * Based on original screenshot layer written by:
 * Author: Cody Northrop <cody@lunarg.com>
 * Author: David Pinedo <david@lunarg.com>
 * Author: Jon Ashburn <jon@lunarg.com>
 */

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unordered_map>
#include <iostream>
#include <algorithm>
#include <list>
#include <map>
#include <vector>
#include <fstream>
#include "png.h"

using namespace std;

#include "vk_layer_platform.h"
#include "vk_dispatch_table_helper.h"
#include "vk_struct_string_helper_cpp.h"
#include "vk_layer_config.h"
#include "vk_layer_table.h"
#include "vk_layer_extension_utils.h"

// Enabled device extensions
struct devExts {
    bool wsi_enabled;
};
static std::unordered_map<void *, struct devExts>     deviceExtMap;
static device_table_map screenshot_device_table_map;
//TODO convert over to the new interface using locally defiend maps
//static instance_table_map screenshot_instance_table_map;

static int globalLockInitialized = 0;
static layer_platform_thread_mutex globalLock;

// unordered map: associates a swap chain with a device, image extent, format, and
// list of images
typedef struct
{
    VkDevice device;
    VkExtent2D imageExtent;
    VkFormat format;
    VkImage *imageList;
} SwapchainMapStruct;
static unordered_map<VkSwapchainKHR, SwapchainMapStruct *> swapchainMap;

// unordered map: associates an image with a device, image extent, and format
typedef struct
{
    VkDevice device;
    VkExtent2D imageExtent;
    VkFormat format;
} ImageMapStruct;
static unordered_map<VkImage, ImageMapStruct *> imageMap;

// unordered map: associates a device with a queue, commandPool, and physical device
typedef struct
{
    VkQueue  queue;
    VkCommandPool commandPool;
    VkPhysicalDevice physicalDevice;
} DeviceMapStruct;
static unordered_map<VkDevice, DeviceMapStruct *> deviceMap;

// unordered map: associates a physical device with an instance
typedef struct
{
    VkInstance instance;
} PhysDeviceMapStruct;
static unordered_map<VkPhysicalDevice, PhysDeviceMapStruct *> physDeviceMap;

// List of frames to we will get a screenshot of
static vector<int> screenshotFrames;

// Flag indicating we have queried _VK_SCREENSHOT env var
static bool screenshotEnvQueried = false;

// Finds a memory type with the specified properties.
// Return false if none exist
static bool memory_type_from_properties(
    VkPhysicalDeviceMemoryProperties *memory_properties,
    uint32_t typeBits,
    VkFlags requirements_mask,
    uint32_t *typeIndex)
{
     // Search memtypes to find first index with those properties
     for (uint32_t i = 0; i < 32; i++) {
         if ((typeBits & 1) == 1) {
             // Type is available, does it match user properties?
             if ((memory_properties->memoryTypes[i].propertyFlags & requirements_mask) == requirements_mask) {
                 *typeIndex = i;
                 return true;
             }
         }
         typeBits >>= 1;
     }
     // No memory types matched, return failure
     return false;
}

// Perform layer-specific initialization
static void init_png_screenshot_layer()
{
    if (!globalLockInitialized)
    {
        // TODO/TBD: Need to delete this mutex sometime.  How???  One
        // suggestion is to call this during vkCreateInstance(), and then we
        // can clean it up during vkDestroyInstance().  However, that requires
        // that the layer have per-instance locks.  We need to come back and
        // address this soon.
        layer_platform_thread_create_mutex(&globalLock);
        globalLockInitialized = 1;
    }
}

// Write a png file with the supplied name, properties, and contents.
static void writePNG(const char *filename, VkFormat format, uint32_t width, uint32_t height, VkDeviceSize alignment, const char* pData)
{
    FILE* hFile = fopen(filename, "wb");
    if (hFile == NULL)
    {
        printf("Failed to open '%s' for writing png file.\n", filename);
    }
    else
    {
        UINT_PTR* pdwRowPtrs = NULL;

        // Create PNG write 'context'
        png_structp pPNG = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        png_infop pPNGInfo = NULL;

        if (pPNG == NULL)
        {
            printf("Failed to create PNG write struct.\n");
        }
        else
        {
            pPNGInfo = png_create_info_struct(pPNG);
        }

        if (pPNGInfo == NULL)
        {
            printf("Failed to create PNG info struct.\n");
        }
        else
        {
            png_init_io(pPNG, hFile);

            // Set header
            png_set_IHDR(pPNG, pPNGInfo, width, height, 8, PNG_COLOR_TYPE_RGBA, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

            // write header and palette info
            png_write_info(pPNG, pPNGInfo);

            png_set_packswap(pPNG);

            // Set up pointers for each row
            VkDeviceSize pitch = (VkDeviceSize)ceil(width * 4.0f / (float)alignment) * alignment;
            pdwRowPtrs = new UINT_PTR[height];
            DWORD dwIndex;
            DWORD x;
            if (format == VK_FORMAT_B8G8R8A8_UNORM)
            {
                UINT_PTR* uiPtr = (UINT_PTR*)pData;
                for (dwIndex = 0; dwIndex < height; dwIndex++)
                {
                    // swizzle the pixels from abgr -> argb
                    unsigned int* pRow = (unsigned int*)&pData[dwIndex*pitch];
                    for (x = 0; x < width; x++)
                    {
                        unsigned int argb = (*pRow & 0xff00ff00) | (*pRow & 0x000000ff) << 16 | (*pRow & 0x00ff0000) >> 16;
                        *pRow = argb;
                        pRow++;
                    }

                    // set the pointer for each row
                    pdwRowPtrs[dwIndex] = (UINT_PTR)pData + (dwIndex * pitch);
                }
            }
            else if (format == VK_FORMAT_R8G8B8A8_UNORM)
            {
                for (dwIndex = 0; dwIndex < height; dwIndex++)
                {
                    pdwRowPtrs[dwIndex] = (UINT_PTR)pData + (dwIndex * pitch);
                }
            }
            else
            {
                // TODO: add support for additional formats
                printf("Unrecognized image format, but will attempt png anyway.\n");
                for (dwIndex = 0; dwIndex < height; dwIndex++)
                {
                    pdwRowPtrs[dwIndex] = (UINT_PTR)pData + (dwIndex * pitch);
                }
            }

            // Write the image
            png_write_image(pPNG, (png_bytep*)pdwRowPtrs);

            // Finish up
            png_write_end(pPNG, pPNGInfo);

            delete[] pdwRowPtrs;
        }

        if (pPNG)
        {
            if (pPNGInfo)
            {
                png_destroy_write_struct(&pPNG, &pPNGInfo);
            }
            else
            {
                png_destroy_write_struct(&pPNG, NULL);
            }
        }

        fclose(hFile);
    }
}

// Generates a PNG file from the contents of a VkImage.
// Converts the supplied VkImage into a resource that can be read back to the CPU,
// then uses that second image to obtain the contents to be written to the PNG file.
static void generate_png_from_vkimage(const char* filename, VkImage image1)
{
    VkImage image2;
    VkResult err;
    bool pass;
    const char *ptr;
    VkDeviceMemory mem2;
    VkCommandBuffer commandBuffer;

    if (imageMap.empty() || imageMap.find(image1) == imageMap.end())
        return;

    VkDevice device = imageMap[image1]->device;
    VkPhysicalDevice physicalDevice = deviceMap[device]->physicalDevice;
    VkInstance instance = physDeviceMap[physicalDevice]->instance;
    VkQueue queue = deviceMap[device]->queue;
    uint32_t width = imageMap[image1]->imageExtent.width;
    uint32_t height = imageMap[image1]->imageExtent.height;
    VkFormat format = imageMap[image1]->format;
    const VkImageSubresource sr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0 };
    VkSubresourceLayout sr_layout;
    const VkImageCreateInfo imgCreateInfo = {
        VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        NULL,
        0,
        VK_IMAGE_TYPE_2D,
        format,
        { width, height, 1 },
        1,
        1,
        VK_SAMPLE_COUNT_1_BIT,
        VK_IMAGE_TILING_LINEAR,
        VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_SHARING_MODE_EXCLUSIVE,
        0,
        NULL,
        VK_IMAGE_LAYOUT_UNDEFINED,
    };
    VkMemoryAllocateInfo memAllocInfo = {
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        NULL,
        0,     // allocationSize, queried later
        0      // memoryTypeIndex, queried later
    };
    const VkCommandBufferAllocateInfo allocCommandBufferInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        NULL,
        deviceMap[device]->commandPool,
        VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        1
    };
    const VkCommandBufferBeginInfo commandBufferBeginInfo = {
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        NULL,
        VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    const VkImageCopy imageCopyRegion = {
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        { 0, 0, 0 },
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        { 0, 0, 0 },
        { width, height, 1 }
    };
    VkMemoryRequirements memRequirements;
    VkLayerDispatchTable* pTableDevice = get_dispatch_table(screenshot_device_table_map, device);
    VkLayerDispatchTable* pTableQueue = get_dispatch_table(screenshot_device_table_map, queue);
    VkLayerInstanceDispatchTable* pInstanceTable;
    VkLayerDispatchTable* pTableCommandBuffer;
    VkPhysicalDeviceMemoryProperties memory_properties;

    // The VkImage image1 we are going to dump may not be mappable,
    // and/or it may have a tiling mode of optimal rather than linear.
    // To make sure we have an image that we can map and read linearly, we:
    //     create image2 that is mappable and linear
    //     copy image1 to image2
    //     map image2
    //     read from image2's mapped memory.

    err = pTableDevice->CreateImage(device, &imgCreateInfo, NULL, &image2);
    assert(!err);

    pTableDevice->GetImageMemoryRequirements(device, image2, &memRequirements);

    memAllocInfo.allocationSize = memRequirements.size;
    pInstanceTable = instance_dispatch_table(instance);
    pInstanceTable->GetPhysicalDeviceMemoryProperties(physicalDevice, &memory_properties);

    pass = memory_type_from_properties(&memory_properties,
        memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        &memAllocInfo.memoryTypeIndex);
    assert(pass);

    err = pTableDevice->AllocateMemory(device, &memAllocInfo, NULL, &mem2);
    assert(!err);

    err = pTableQueue->BindImageMemory(device, image2, mem2, 0);
    assert(!err);

    err = pTableDevice->AllocateCommandBuffers(device, &allocCommandBufferInfo, &commandBuffer);
    assert(!err);

    screenshot_device_table_map.emplace(commandBuffer, pTableDevice);
    pTableCommandBuffer = screenshot_device_table_map[commandBuffer];

    // We have just created a dispatchable object, but the dispatch table has not been placed
    // in the object yet.  When a "normal" application creates a command buffer, the dispatch
    // table is installed by the top-level api binding (trampoline.c).
    // But here, we have to do it ourselves.
    *((const void**)commandBuffer) = *(void**)device;

    err = pTableCommandBuffer->BeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
    assert(!err);

    // Transition the source image layout to prepare it for the copy.

    VkImageMemoryBarrier image1_memory_barrier = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        NULL,
        0,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        0,
        0,
        image1,
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    };

    VkImageMemoryBarrier image2_memory_barrier = {
        VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        NULL,
        0,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL,
        0,
        0,
        image2,
        { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 }
    };

    VkPipelineStageFlags src_stages = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkPipelineStageFlags dst_stages = VK_PIPELINE_STAGE_TRANSFER_BIT;

    pTableCommandBuffer->CmdPipelineBarrier(commandBuffer, src_stages, dst_stages,
        0, 0, NULL, 0, NULL, 1, &image1_memory_barrier);

    pTableCommandBuffer->CmdCopyImage(commandBuffer,
        image1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        image2, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        1, &imageCopyRegion);

    pTableCommandBuffer->CmdPipelineBarrier(commandBuffer, src_stages, dst_stages,
        0, 0, NULL, 0, NULL, 1, &image2_memory_barrier);

    err = pTableCommandBuffer->EndCommandBuffer(commandBuffer);
    assert(!err);

    VkFence nullFence = { VK_NULL_HANDLE };
    VkSubmitInfo submit_info;
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = NULL;
    submit_info.waitSemaphoreCount = 0;
    submit_info.pWaitSemaphores = NULL;
    submit_info.pWaitDstStageMask = NULL;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &commandBuffer;
    submit_info.signalSemaphoreCount = 0;
    submit_info.pSignalSemaphores = NULL;

    err = pTableQueue->QueueSubmit(queue, 1, &submit_info, nullFence);
    assert(!err);

    err = pTableQueue->QueueWaitIdle(queue);
    assert(!err);

    err = pTableDevice->DeviceWaitIdle(device);
    assert(!err);

    pTableDevice->GetImageSubresourceLayout(device, image2, &sr, &sr_layout);

    err = pTableDevice->MapMemory(device, mem2, 0, 0, 0, (void **)&ptr);
    assert(!err);

    ptr += sr_layout.offset;

    // Write the PNG file to disk
    writePNG(filename, format, width, height, memRequirements.alignment, ptr);

    pTableDevice->UnmapMemory(device, mem2);

    // Restore the swap chain image layout to what it was before.
    // This may not be strictly needed, but it is generally good to restore things to original state.
    err = pTableCommandBuffer->BeginCommandBuffer(commandBuffer, &commandBufferBeginInfo);
    assert(!err);
    image1_memory_barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    image1_memory_barrier.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    pTableCommandBuffer->CmdPipelineBarrier(commandBuffer, src_stages, dst_stages,
        0, 0, NULL, 0, NULL, 1, &image1_memory_barrier);
    err = pTableCommandBuffer->EndCommandBuffer(commandBuffer);
    assert(!err);

    err = pTableQueue->QueueSubmit(queue, 1, &submit_info, nullFence);
    assert(!err);

    err = pTableQueue->QueueWaitIdle(queue);
    assert(!err);

    err = pTableDevice->DeviceWaitIdle(device);
    assert(!err);

    // Clean up
    pTableDevice->FreeMemory(device, mem2, NULL);
    pTableDevice->FreeCommandBuffers(device, deviceMap[device]->commandPool, 1, &commandBuffer);
}

// Called when the application creates an instance and this layer is enabled.
// All layers must query the next layer to call and then initialize the instance table.
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkInstance* pInstance)
{
    VkLayerInstanceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);

    assert(chain_info->u.pLayerInfo);
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    assert(fpGetInstanceProcAddr);
    PFN_vkCreateInstance fpCreateInstance = (PFN_vkCreateInstance) fpGetInstanceProcAddr(*pInstance, "vkCreateInstance");
    if (fpCreateInstance == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = fpCreateInstance(pCreateInfo, pAllocator, pInstance);
    if (result != VK_SUCCESS)
        return result;

    initInstanceTable(*pInstance, fpGetInstanceProcAddr);

    init_png_screenshot_layer();

    return result;
}

//TODO hook DestroyInstance to cleanup

// Called when a layer is being enabled.
// All layers must hook the entrypoints that they are interested in monitoring.
static void createDeviceRegisterExtensions(const VkDeviceCreateInfo* pCreateInfo, VkDevice device)
{
    uint32_t i;
    VkLayerDispatchTable *pDisp  = get_dispatch_table(screenshot_device_table_map, device);
    PFN_vkGetDeviceProcAddr gpa = pDisp->GetDeviceProcAddr;
    pDisp->CreateSwapchainKHR = (PFN_vkCreateSwapchainKHR) gpa(device, "vkCreateSwapchainKHR");
    pDisp->GetSwapchainImagesKHR = (PFN_vkGetSwapchainImagesKHR) gpa(device, "vkGetSwapchainImagesKHR");
    pDisp->AcquireNextImageKHR = (PFN_vkAcquireNextImageKHR) gpa(device, "vkAcquireNextImageKHR");
    pDisp->QueuePresentKHR = (PFN_vkQueuePresentKHR) gpa(device, "vkQueuePresentKHR");
    deviceExtMap[pDisp].wsi_enabled = false;
    for (i = 0; i < pCreateInfo->enabledExtensionCount; i++) {
        if (strcmp(pCreateInfo->ppEnabledExtensionNames[i], VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0)
            deviceExtMap[pDisp].wsi_enabled = true;
    }
}

// Called when the application creates a device and this layer is enabled.
// All layers must query the next layer to call and then initialize the device table.
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateDevice(
    VkPhysicalDevice          gpu,
    const VkDeviceCreateInfo *pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDevice                 *pDevice)
{
    VkLayerDeviceCreateInfo *chain_info = get_chain_info(pCreateInfo, VK_LAYER_LINK_INFO);

    assert(chain_info->u.pLayerInfo);
    PFN_vkGetInstanceProcAddr fpGetInstanceProcAddr = chain_info->u.pLayerInfo->pfnNextGetInstanceProcAddr;
    PFN_vkGetDeviceProcAddr fpGetDeviceProcAddr = chain_info->u.pLayerInfo->pfnNextGetDeviceProcAddr;
    PFN_vkCreateDevice fpCreateDevice = (PFN_vkCreateDevice) fpGetInstanceProcAddr(NULL, "vkCreateDevice");
    if (fpCreateDevice == NULL) {
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    // Advance the link info for the next element on the chain
    chain_info->u.pLayerInfo = chain_info->u.pLayerInfo->pNext;

    VkResult result = fpCreateDevice(gpu, pCreateInfo, pAllocator, pDevice);
    if (result != VK_SUCCESS) {
        return result;
    }

    initDeviceTable(*pDevice, fpGetDeviceProcAddr, screenshot_device_table_map);

    createDeviceRegisterExtensions(pCreateInfo, *pDevice);
    // Create a mapping from a device to a physicalDevice
    if (deviceMap[*pDevice] == NULL)
    {
        DeviceMapStruct *deviceMapElem = new DeviceMapStruct;
        deviceMap[*pDevice] = deviceMapElem;
    }
    deviceMap[*pDevice]->physicalDevice = gpu;

    return result;
}

// Called when the application enumerates physical devices.
// In order to take screenshots, we need to keep track of which physical devices
// are associated with each Vulkan instance.
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumeratePhysicalDevices(
    VkInstance instance,
    uint32_t* pPhysicalDeviceCount,
    VkPhysicalDevice* pPhysicalDevices)
{
    VkResult result;

    VkLayerInstanceDispatchTable* pTable = instance_dispatch_table(instance);
    result = pTable->EnumeratePhysicalDevices(instance, pPhysicalDeviceCount, pPhysicalDevices);
    if (result==VK_SUCCESS && *pPhysicalDeviceCount > 0 && pPhysicalDevices)
    {
        for (uint32_t i=0; i<*pPhysicalDeviceCount ; i++)
        {
            // Create a mapping from a physicalDevice to an instance
            if (physDeviceMap[pPhysicalDevices[i]] == NULL)
            {
                PhysDeviceMapStruct *physDeviceMapElem = new PhysDeviceMapStruct;
                physDeviceMap[pPhysicalDevices[i]] = physDeviceMapElem;
            }
            physDeviceMap[pPhysicalDevices[i]]->instance = instance;
        }
    }
    return result;
}

/* TODO: Probably need a DestroyDevice as well */

// Properties of the png screenshot layer.
// Includes name, the API version the layer is written against, a layer implementation version, and a description.
static const VkLayerProperties ss_device_layers[] = {
    {
        "VK_LAYER_AMD_png_screenshot",
        VK_API_VERSION,
        1,
        "Layer: png screenshot",
    }
};

// Called to find out if this layer implements any API extensions to the Vulkan instance.
// PNG screenshot layer does not.
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
        const char *pLayerName,
        uint32_t *pCount,
        VkExtensionProperties* pProperties)
{
    /* ScreenShot does not have any global extensions */
    return util_GetExtensionProperties(0, NULL, pCount, pProperties);
}

// Called to enumerate properties of Vulkan instance layers.
// PNG screenshot doesn't have any to add.
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
        uint32_t *pCount,
        VkLayerProperties*    pProperties)
{
    /* ScreenShot does not have any global layers */
    return util_GetLayerProperties(0, NULL,
                                   pCount, pProperties);
}

// Called to enumerate device-specific extensions.
// PNG Screenshot doesn't implement one.
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceExtensionProperties(
        VkPhysicalDevice                            physicalDevice,
        const char*                                 pLayerName,
        uint32_t*                                   pCount,
        VkExtensionProperties*                      pProperties)
{
    /* ScreenShot does not have any physical device extensions */
    if (pLayerName == NULL) {
        VkLayerInstanceDispatchTable* pTable = instance_dispatch_table(physicalDevice);
        return pTable->EnumerateDeviceExtensionProperties(
                                                    physicalDevice,
                                                    NULL,
                                                    pCount,
                                                    pProperties);
    } else {
        return util_GetExtensionProperties(0, NULL, pCount, pProperties);
    }
}

// Called to enumerate device-specific layer properties.
// return the PNG screenshot layer properties (ie: name, api version, layer version, description)
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateDeviceLayerProperties(
        VkPhysicalDevice                            physicalDevice,
        uint32_t*                                   pCount,
        VkLayerProperties*                          pProperties)
{
    return util_GetLayerProperties(ARRAY_SIZE(ss_device_layers),
                                   ss_device_layers,
                                   pCount, pProperties);
}

// Called when the application wants to get a device queue.
// PNG screenshot needs to track which queues are on each device.
VK_LAYER_EXPORT VKAPI_ATTR void VKAPI_CALL vkGetDeviceQueue(
    VkDevice  device,
    uint32_t  queueNodeIndex,
    uint32_t  queueIndex,
    VkQueue   *pQueue)
{
    VkLayerDispatchTable* pTable = screenshot_device_table_map[device];
    get_dispatch_table(screenshot_device_table_map, device)->GetDeviceQueue(device, queueNodeIndex, queueIndex, pQueue);

    layer_platform_thread_lock_mutex(&globalLock);
    if (screenshotEnvQueried && screenshotFrames.empty()) {
        // We are all done taking screenshots, so don't do anything else
        layer_platform_thread_unlock_mutex(&globalLock);
        return;
    }

    screenshot_device_table_map.emplace(*pQueue, pTable);

    // Create a mapping from a device to a queue
    if (deviceMap[device] == NULL)
    {
        DeviceMapStruct *deviceMapElem = new DeviceMapStruct;
        deviceMap[device] = deviceMapElem;
    }
    deviceMap[device]->queue = *pQueue;
    layer_platform_thread_unlock_mutex(&globalLock);
}

// Called when the application wants to create a command pool.
// PNG screenshot needs to track which command pool is on each device.
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateCommandPool(
    VkDevice  device,
    const VkCommandPoolCreateInfo *pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkCommandPool *pCommandPool)
{
    VkLayerDispatchTable* pTable = screenshot_device_table_map[device];
    VkResult result = get_dispatch_table(screenshot_device_table_map, device)->CreateCommandPool(device, pCreateInfo, pAllocator, pCommandPool);

    layer_platform_thread_lock_mutex(&globalLock);
    if (screenshotEnvQueried && screenshotFrames.empty()) {
        // We are all done taking screenshots, so don't do anything else
        layer_platform_thread_unlock_mutex(&globalLock);
        return result;
    }

    // Create a mapping from a device to a commandPool
    if (deviceMap[device] == NULL)
    {
        DeviceMapStruct *deviceMapElem = new DeviceMapStruct;
        deviceMap[device] = deviceMapElem;
    }
    deviceMap[device]->commandPool = *pCommandPool;
    layer_platform_thread_unlock_mutex(&globalLock);
    return result;
}

// Called when the application wants to create a swapchain.
// PNG screenshot needs to track swapchain information on each device
// so it knows the appropriate dimensions and pixel format.
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateSwapchainKHR(
    VkDevice                        device,
    const VkSwapchainCreateInfoKHR *pCreateInfo,
    const VkAllocationCallbacks    *pAllocator,
    VkSwapchainKHR                 *pSwapchain)
{
    VkLayerDispatchTable* pTable = screenshot_device_table_map[device];
    VkResult result = get_dispatch_table(screenshot_device_table_map, device)->CreateSwapchainKHR(device, pCreateInfo, pAllocator, pSwapchain);

    layer_platform_thread_lock_mutex(&globalLock);
    if (screenshotEnvQueried && screenshotFrames.empty()) {
        // We are all done taking screenshots, so don't do anything else
        layer_platform_thread_unlock_mutex(&globalLock);
        return result;
    }

    if (result == VK_SUCCESS)
    {
        // Create a mapping for a swapchain to a device, image extent, and format
        SwapchainMapStruct *swapchainMapElem = new SwapchainMapStruct;
        swapchainMapElem->device = device;
        swapchainMapElem->imageExtent = pCreateInfo->imageExtent;
        swapchainMapElem->format = pCreateInfo->imageFormat;
        swapchainMap.insert(make_pair(*pSwapchain, swapchainMapElem));

        // Create a mapping for the swapchain object into the dispatch table
        screenshot_device_table_map.emplace((void *)pSwapchain, pTable);
    }
    layer_platform_thread_unlock_mutex(&globalLock);

    return result;
}

// Called when the application wants to get the swapchain images.
// PNG screenshot needs to know how many images are on each swapchain
// so that it can query their content.
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkGetSwapchainImagesKHR(
    VkDevice                device,
    VkSwapchainKHR          swapchain,
    uint32_t*               pCount,
    VkImage*                pSwapchainImages)
{
    VkResult result = get_dispatch_table(screenshot_device_table_map, device)->GetSwapchainImagesKHR(device, swapchain, pCount, pSwapchainImages);

    layer_platform_thread_lock_mutex(&globalLock);
    if (screenshotEnvQueried && screenshotFrames.empty()) {
        // We are all done taking screenshots, so don't do anything else
        layer_platform_thread_unlock_mutex(&globalLock);
        return result;
    }

    if (result == VK_SUCCESS &&
        pSwapchainImages &&
        !swapchainMap.empty() && swapchainMap.find(swapchain) != swapchainMap.end())
    {
        unsigned i;

        for (i=0; i<*pCount; i++)
        {
            // Create a mapping for an image to a device, image extent, and format
            if (imageMap[pSwapchainImages[i]] == NULL)
            {
                ImageMapStruct *imageMapElem = new ImageMapStruct;
                imageMap[pSwapchainImages[i]] = imageMapElem;
            }
            imageMap[pSwapchainImages[i]]->device = swapchainMap[swapchain]->device;
            imageMap[pSwapchainImages[i]]->imageExtent = swapchainMap[swapchain]->imageExtent;
            imageMap[pSwapchainImages[i]]->format = swapchainMap[swapchain]->format;
        }

        // Add list of images to swapchain to image map
        SwapchainMapStruct *swapchainMapElem = swapchainMap[swapchain];
        if (i >= 1 && swapchainMapElem)
        {
            VkImage *imageList = new VkImage[i];
            swapchainMapElem->imageList = imageList;
            for (unsigned j=0; j<i; j++)
            {
                swapchainMapElem->imageList[j] = pSwapchainImages[j];
            }
        }

    }   
    layer_platform_thread_unlock_mutex(&globalLock);
    return result;
}

static void parse_screenshot_env()
{

    const char *_vk_screenshot = layer_platform_getenv("_VK_PNG_SCREENSHOT");
    if (_vk_screenshot && *_vk_screenshot)
    {
        int startFrame = 0;
        int endFrame = 0;
        int stepFrame = 0;
        if (sscanf(_vk_screenshot, "%d-%d,%d", &startFrame, &endFrame, &stepFrame) == 3)
        {
            // A range of frames, and a step count.
            for (int frame = startFrame; frame <= endFrame; frame += stepFrame)
            {
                screenshotFrames.push_back(frame);
            }
        }
        else
        {
            // A comma-separated list of frame numbers

            string spec(_vk_screenshot), word;
            size_t start = 0, comma = 0;

            while (start < spec.size())
            {
                int frameToAdd;
                comma = spec.find(',', start);
                if (comma == string::npos)
                {
                    word = string(spec, start);
                }
                else
                {
                    word = string(spec, start, comma - start);
                }
                frameToAdd = atoi(word.c_str());
                // Add the frame number to list, but only do it if the word started with a digit and if
                // it's not already in the list
                if (*(word.c_str()) >= '0' &&
                    *(word.c_str()) <= '9' &&
                    find(screenshotFrames.begin(), screenshotFrames.end(), frameToAdd) == screenshotFrames.end())
                {
                    screenshotFrames.push_back(frameToAdd);
                }
                if (comma == string::npos)
                {
                    break;
                }
                start = comma + 1;
            }
        }
    }
    layer_platform_free_getenv(_vk_screenshot);
}

// Called to present the pending Vulkan swapchain images.
// During this call, the PNG Screenshot layer checks to see if the frame number matches 
// one of the requested frames to capture, and saves the image to a PNG file if needed.
VK_LAYER_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkQueuePresentKHR(VkQueue queue, const VkPresentInfoKHR* pPresentInfo)
{
    static int frameNumber = 0;
    if (frameNumber == 10) {fflush(stdout); /* *((int*)0)=0; */ }

    // Pass the vkQueuePresentKHR call down to the next layer
    VkResult result = get_dispatch_table(screenshot_device_table_map, queue)->QueuePresentKHR(queue, pPresentInfo);

    layer_platform_thread_lock_mutex(&globalLock);

    // If they env var hasn't already been parsed, then parse it now
    // so the layer knows which frames to save to file.
    if (!screenshotEnvQueried)
    {
        parse_screenshot_env();
        screenshotEnvQueried = true;
    }
    
    if (result == VK_SUCCESS && !screenshotFrames.empty())
    {
        vector<int>::iterator it;
        it = find(screenshotFrames.begin(), screenshotFrames.end(), frameNumber);
        if (it != screenshotFrames.end())
        {
            string fileName;
            fileName = to_string(frameNumber) + ".png";

            VkImage image;
            VkSwapchainKHR swapchain;
            // We'll dump only one image: the first
            swapchain = pPresentInfo->pSwapchains[0];
            image = swapchainMap[swapchain]->imageList[pPresentInfo->pImageIndices[0]];
            generate_png_from_vkimage(fileName.c_str(), image);
            screenshotFrames.erase(it);

            if (screenshotFrames.empty())
            {
                // Free all our maps since we are done with them.
                for (auto it = swapchainMap.begin(); it != swapchainMap.end(); it++)
                {
                    SwapchainMapStruct *swapchainMapElem = it->second;
                    delete swapchainMapElem;
                }
                for (auto it = imageMap.begin(); it != imageMap.end(); it++)
                {
                    ImageMapStruct *imageMapElem = it->second;
                    delete imageMapElem;
                }
                for (auto it = deviceMap.begin(); it != deviceMap.end(); it++)
                {
                    DeviceMapStruct *deviceMapElem = it->second;
                    delete deviceMapElem;
                }
                for (auto it = physDeviceMap.begin(); it != physDeviceMap.end(); it++)
                {
                    PhysDeviceMapStruct *physDeviceMapElem = it->second;
                    delete physDeviceMapElem;
                }
                swapchainMap.clear();
                imageMap.clear();
                deviceMap.clear();
                physDeviceMap.clear();
            }
        }
    }

    frameNumber++;
    layer_platform_thread_unlock_mutex(&globalLock);
    return result;
}

// Called to query a device proc addr from this layer.
// Since we want to intercept a few entrypoints, we need to return our local functions.
VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetDeviceProcAddr(
    VkDevice         dev,
    const char       *funcName)
{
    if (strcmp(funcName, "vkGetDeviceProcAddr") == 0)
        return (PFN_vkVoidFunction)vkGetDeviceProcAddr;
    if (strcmp(funcName, "vkGetDeviceQueue") == 0)
        return (PFN_vkVoidFunction) vkGetDeviceQueue;
    if (strcmp(funcName, "vkCreateCommandPool") == 0)
        return (PFN_vkVoidFunction) vkCreateCommandPool;

    if (dev == NULL) {
        return NULL;
    }

    VkLayerDispatchTable *pDisp =  get_dispatch_table(screenshot_device_table_map, dev);
    if (deviceExtMap.size() != 0 && deviceExtMap[pDisp].wsi_enabled)
    {
        if (!strcmp(funcName, "vkCreateSwapchainKHR"))
            return (PFN_vkVoidFunction) vkCreateSwapchainKHR;
        if (!strcmp(funcName, "vkGetSwapchainImagesKHR"))
            return (PFN_vkVoidFunction) vkGetSwapchainImagesKHR;
        if (!strcmp(funcName, "vkQueuePresentKHR"))
            return (PFN_vkVoidFunction) vkQueuePresentKHR;
    }

    if (pDisp->GetDeviceProcAddr == NULL)
        return NULL;
    return pDisp->GetDeviceProcAddr(dev, funcName);
}

// Called to query an instance proc addr from this layer.
// Since we want to intercept a few entrypoints, we need to return our local functions.
VK_LAYER_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* funcName)
{
    if (strcmp("vkGetInstanceProcAddr", funcName) == 0)
        return (PFN_vkVoidFunction) vkGetInstanceProcAddr;
    if (strcmp(funcName, "vkCreateInstance") == 0)
        return (PFN_vkVoidFunction) vkCreateInstance;
    if (strcmp(funcName, "vkCreateDevice") == 0)
        return (PFN_vkVoidFunction) vkCreateDevice;
    if (strcmp(funcName, "vkEnumeratePhysicalDevices") == 0)
        return (PFN_vkVoidFunction)vkEnumeratePhysicalDevices;
    if (strcmp(funcName, "vkEnumerateDeviceExtensionProperties") == 0)
        return (PFN_vkVoidFunction)vkEnumerateDeviceExtensionProperties;

    if (instance == VK_NULL_HANDLE) {
        return NULL;
    }
    VkLayerInstanceDispatchTable* pTable = instance_dispatch_table(instance);
    if (pTable->GetInstanceProcAddr == NULL)
        return NULL;
    return pTable->GetInstanceProcAddr(instance, funcName);
}
