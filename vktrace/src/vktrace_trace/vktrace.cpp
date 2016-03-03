/**************************************************************************
 *
 * Copyright 2014-2016 Valve Corporation
 * Copyright (C) 2014-2016 LunarG, Inc.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * Author: Jon Ashburn <jon@lunarg.com>
 * Author: Peter Lohrmann <peterl@valvesoftware.com>
 **************************************************************************/
#include "vktrace.h"

#include "vktrace_process.h"

extern "C" {
#include "vktrace_common.h"
#include "vktrace_filelike.h"
#include "vktrace_interconnect.h"
#include "vktrace_trace_packet_identifiers.h"
#include "vktrace_trace_packet_utils.h"
}

#include <sys/types.h>
#include <sys/stat.h>

vktrace_settings g_settings;
vktrace_settings g_default_settings;

vktrace_SettingInfo g_settings_info[] =
{
    // common command options
    { "p", "Program", VKTRACE_SETTING_STRING, &g_settings.program, &g_default_settings.program, TRUE, "The program to trace."},
    { "a", "Arguments", VKTRACE_SETTING_STRING, &g_settings.arguments, &g_default_settings.arguments, TRUE, "Cmd-line arguments to pass to trace program."},
    { "w", "WorkingDir", VKTRACE_SETTING_STRING, &g_settings.working_dir, &g_default_settings.working_dir, TRUE, "The program's working directory."},
    { "o", "OutputTrace", VKTRACE_SETTING_STRING, &g_settings.output_trace, &g_default_settings.output_trace, TRUE, "Path to the generated output trace file."},
    { "s", "ScreenShot", VKTRACE_SETTING_STRING, &g_settings.screenshotList, &g_default_settings.screenshotList, TRUE, "Comma separated list of frame numbers on which to take a screen snapshot."},
    { "png", "PngScreenShot", VKTRACE_SETTING_STRING, &g_settings.pngScreenshotList, &g_default_settings.pngScreenshotList, TRUE, "Saves a PNG screenshot of frames identified by: <startFrame>-<endFrame>,<stepFrames>." },
    { "ptm", "PrintTraceMessages", VKTRACE_SETTING_BOOL, &g_settings.print_trace_messages, &g_default_settings.print_trace_messages, TRUE, "Print trace messages to vktrace console."},

    //{ "z", "pause", VKTRACE_SETTING_BOOL, &g_settings.pause, &g_default_settings.pause, TRUE, "Wait for a key at startup (so a debugger can be attached)" },
    //{ "q", "quiet", VKTRACE_SETTING_BOOL, &g_settings.quiet, &g_default_settings.quiet, TRUE, "Disable warning, verbose, and debug output" },
    //{ "v", "verbose", VKTRACE_SETTING_BOOL, &g_settings.verbose, &g_default_settings.verbose, TRUE, "Enable verbose output" },
    //{ "d", "debug", VKTRACE_SETTING_BOOL, &g_settings.debug, &g_default_settings.debug, TRUE, "Enable verbose debug information" },
};

vktrace_SettingGroup g_settingGroup =
{
    "vktrace",
    sizeof(g_settings_info) / sizeof(g_settings_info[0]),
    &g_settings_info[0]
};

// ------------------------------------------------------------------------------------------------
#if defined(WIN32)
void MessageLoop()
{
    MSG msg = { 0 };
    bool quit = false;
    while (!quit)
    {
        if (GetMessage(&msg, NULL, 0, 0) == FALSE)
        {
            quit = true;
        }
        else
        {
            quit = (msg.message == VKTRACE_WM_COMPLETE);
        }
    }
}
#endif

int PrepareTracers(vktrace_process_capture_trace_thread_info** ppTracerInfo)
{
    unsigned int num_tracers = 1;

    assert(ppTracerInfo != NULL && *ppTracerInfo == NULL);
    *ppTracerInfo = VKTRACE_NEW_ARRAY(vktrace_process_capture_trace_thread_info, num_tracers);
    memset(*ppTracerInfo, 0, sizeof(vktrace_process_capture_trace_thread_info) * num_tracers);

    // we only support Vulkan tracer
    (*ppTracerInfo)[0].tracerId = VKTRACE_TID_VULKAN;

    return num_tracers;
}

bool InjectTracersIntoProcess(vktrace_process_info* pInfo)
{
    bool bRecordingThreadsCreated = true;
    vktrace_thread tracingThread;
    if (vktrace_platform_remote_load_library(pInfo->hProcess, NULL, &tracingThread, NULL)) {
        // prepare data for capture threads
        pInfo->pCaptureThreads[0].pProcessInfo = pInfo;
        pInfo->pCaptureThreads[0].recordingThread = VKTRACE_NULL_THREAD;

        // create thread to record trace packets from the tracer
        pInfo->pCaptureThreads[0].recordingThread = vktrace_platform_create_thread(Process_RunRecordTraceThread, &(pInfo->pCaptureThreads[0]));
        if (pInfo->pCaptureThreads[0].recordingThread == VKTRACE_NULL_THREAD) {
            vktrace_LogError("Failed to create trace recording thread.");
            bRecordingThreadsCreated = false;
        }

    } else {
        // failed to inject a DLL
        bRecordingThreadsCreated = false;
    }
    return bRecordingThreadsCreated;
}

void loggingCallback(VktraceLogLevel level, const char* pMessage)
{
    switch(level)
    {
    case VKTRACE_LOG_ALWAYS: printf("%s\n", pMessage); break;
    case VKTRACE_LOG_DEBUG: printf("Debug: %s\n", pMessage); break;
    case VKTRACE_LOG_ERROR: printf("Error: %s\n", pMessage); break;
    case VKTRACE_LOG_WARNING: printf("Warning: %s\n", pMessage); break;
    case VKTRACE_LOG_VERBOSE: printf("Verbose: %s\n", pMessage); break;
    default:
        printf("%s\n", pMessage); break;
    }

#if defined(WIN32)
#if _DEBUG
    OutputDebugString(pMessage);
#endif
#endif
}

void add_instance_layer(const char* fullLayerName)
{
    char *instanceEnv = vktrace_get_global_var("VK_INSTANCE_LAYERS");
    if (!instanceEnv || strlen(instanceEnv) == 0)
    {
        vktrace_set_global_var("VK_INSTANCE_LAYERS", fullLayerName);
    }
    else if (instanceEnv != strstr(instanceEnv, fullLayerName))
    {
        char *newEnv = vktrace_copy_and_append(fullLayerName, VKTRACE_LIST_SEPARATOR, instanceEnv);
        vktrace_set_global_var("VK_INSTANCE_LAYERS", newEnv);
    }
}

void add_device_layer(const char* fullLayerName)
{
    char *deviceEnv = vktrace_get_global_var("VK_DEVICE_LAYERS");
    if (!deviceEnv || strlen(deviceEnv) == 0)
    {
        vktrace_set_global_var("VK_DEVICE_LAYERS", fullLayerName);
    }
    else if (deviceEnv != strstr(deviceEnv, fullLayerName))
    {
        char *newEnv = vktrace_copy_and_append(fullLayerName, VKTRACE_LIST_SEPARATOR, deviceEnv);
        vktrace_set_global_var("VK_DEVICE_LAYERS", newEnv);
    }
}

// ------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    memset(&g_settings, 0, sizeof(vktrace_settings));

    vktrace_LogSetCallback(loggingCallback);
    vktrace_LogSetLevel(VKTRACE_LOG_LEVEL_MAXIMUM);

    // get vktrace binary directory
    char* execDir = vktrace_platform_get_current_executable_directory();

    // setup defaults
    memset(&g_default_settings, 0, sizeof(vktrace_settings));
    g_default_settings.output_trace = vktrace_copy_and_append(execDir, VKTRACE_PATH_SEPARATOR, "vktrace_out.vktrace");
    g_default_settings.print_trace_messages = FALSE;
    g_default_settings.screenshotList = NULL;
    g_default_settings.pngScreenshotList = NULL;

    // free binary directory string
    vktrace_free(execDir);

    if (vktrace_SettingGroup_init(&g_settingGroup, NULL, argc, argv, &g_settings.arguments) != 0)
    {
        // invalid cmd-line parameters
        vktrace_SettingGroup_delete(&g_settingGroup);
        vktrace_free(g_default_settings.output_trace);
        return -1;
    }
    else
    {
        // Validate vktrace inputs
        BOOL validArgs = TRUE;

        if (g_settings.output_trace == NULL || strlen (g_settings.output_trace) == 0)
        {
            vktrace_LogError("No output trace file (-o) parameter found: Please specify a valid trace file to generate.");
            validArgs = FALSE;
        }
        else
        {
            size_t len = strlen(g_settings.output_trace);
            if (strncmp(&g_settings.output_trace[len-8], ".vktrace", 8) != 0)
            {
                // output trace filename does not end in .vktrace
                vktrace_LogError("Output trace file specified with -o parameter must have a '.vktrace' extension.");
                validArgs = FALSE;
            }
        }

        if (validArgs == FALSE)
        {
            vktrace_SettingGroup_print(&g_settingGroup);
            return -1;
        }

        if (g_settings.program == NULL || strlen(g_settings.program) == 0)
        {
            vktrace_LogAlways("No program (-p) parameter found: Running vktrace as server.");
            g_settings.arguments = NULL;
        }
        else
        {
            if (g_settings.working_dir == NULL || strlen(g_settings.working_dir) == 0)
            {
                CHAR* buf = VKTRACE_NEW_ARRAY(CHAR, 4096);
                vktrace_LogWarning("No working directory (-w) parameter found: Assuming executable's path as working directory.");
                vktrace_platform_full_path(g_settings.program, 4096, buf);
                g_settings.working_dir = vktrace_platform_extract_path(buf);
                VKTRACE_DELETE(buf);
            }

            vktrace_LogAlways("Running vktrace as parent process will spawn child process: %s", g_settings.program);
            if (g_settings.arguments != NULL && strlen(g_settings.arguments) > 0)
            {
                vktrace_LogAlways("Args to be passed to child process: '%s'", g_settings.arguments);
            }
        }
    }

    if (g_settings.screenshotList != NULL)
    {
        // Set env var that communicates list to ScreenShot layer
        vktrace_set_global_var("_VK_SCREENSHOT", g_settings.screenshotList);
    }
    else
    {
        vktrace_set_global_var("_VK_SCREENSHOT", "");
    }

    if (g_settings.pngScreenshotList != NULL)
    {
        unsigned int startFrame = 0;
        unsigned int endFrame = 0;
        unsigned int stepFrame = 0;
        if (sscanf(g_settings.pngScreenshotList, "%u-%u,%u", &startFrame, &endFrame, &stepFrame) == 3)
        {
            // Validate supplying a range of frames, and a step count.
            // example 1: every frame between 10 and 100: "10-100,1"
            // example 2: every 2nd frame between 10 and 100: "10-100,2"
            if (startFrame > endFrame)
            {
                vktrace_LogError("Screenshot start frame (%u) must come BEFORE the end frame (%u).", startFrame, endFrame);
                return 1;
            }

            // set env var that communicates with the PNG ScreenShot layer
            vktrace_set_global_var("_VK_PNG_SCREENSHOT", g_settings.pngScreenshotList);
        }
        else
        {
            vktrace_LogError("PNG Screenshot option must be formatted as: \"<startFrame>-<endFrame>,<stepFrames>\".");
            return 1;
        }
    }
    else
    {
        vktrace_set_global_var("_VK_PNG_SCREENSHOT", "");
    }
    unsigned int serverIndex = 0;
    do {
        // Create and start the process or run in server mode

        BOOL procStarted = TRUE;
        vktrace_process_info procInfo;
        memset(&procInfo, 0, sizeof(vktrace_process_info));
        if (g_settings.program != NULL)
        {
            procInfo.exeName = vktrace_allocate_and_copy(g_settings.program);
            procInfo.processArgs = vktrace_allocate_and_copy(g_settings.arguments);
            procInfo.fullProcessCmdLine = vktrace_copy_and_append(g_settings.program, " ", g_settings.arguments);
            procInfo.workingDirectory = vktrace_allocate_and_copy(g_settings.working_dir);
            procInfo.traceFilename = vktrace_allocate_and_copy(g_settings.output_trace);
        }
        else
        {
            char *pExtension = strrchr(g_settings.output_trace, '.');
            char *basename = vktrace_allocate_and_copy_n(g_settings.output_trace, (int) ((pExtension == NULL) ? strlen(g_settings.output_trace) : pExtension - g_settings.output_trace));
            char num[16];
#ifdef PLATFORM_LINUX
            snprintf(num, 16, "%u", serverIndex);
#elif defined(WIN32)
            _snprintf_s(num, 16, _TRUNCATE, "%u", serverIndex);
#endif
            procInfo.traceFilename = vktrace_copy_and_append(basename, num, pExtension);
         }

        procInfo.parentThreadId = vktrace_platform_get_thread_id();

        // setup tracer, only Vulkan tracer supported
        PrepareTracers(&procInfo.pCaptureThreads);

        if (g_settings.program != NULL)
        {
            // Add PNG ScreenShot layer if enabled
            if (g_settings.pngScreenshotList != NULL)
            {
                add_instance_layer("VK_LAYER_AMD_png_screenshot");
                add_device_layer("VK_LAYER_AMD_png_screenshot");
            }
            // Add ScreenShot layer if enabled
            if (g_settings.screenshotList != NULL)
            {
                add_instance_layer("VK_LAYER_LUNARG_screenshot");
                add_device_layer("VK_LAYER_LUNARG_screenshot");
            }

            // Add vktrace_layer enable env var if needed
            add_instance_layer("VK_LAYER_LUNARG_vktrace");
            add_device_layer("VK_LAYER_LUNARG_vktrace");

            // call CreateProcess to launch the application
            procStarted = vktrace_process_spawn(&procInfo);
        }
        if (procStarted == FALSE)
        {
            vktrace_LogError("Failed to setup remote process.");
        }
        else
        {
            if (InjectTracersIntoProcess(&procInfo) == FALSE)
            {
                vktrace_LogError("Failed to setup tracer communication threads.");
                return -1;
            }

            // create watchdog thread to monitor existence of remote process
            if (g_settings.program != NULL)
                procInfo.watchdogThread = vktrace_platform_create_thread(Process_RunWatchdogThread, &procInfo);

#if defined(PLATFORM_LINUX)
            // Sync wait for local threads and remote process to complete.

            vktrace_platform_sync_wait_for_thread(&(procInfo.pCaptureThreads[0].recordingThread));

            if (g_settings.program != NULL)
                vktrace_platform_sync_wait_for_thread(&procInfo.watchdogThread);
#else
            vktrace_platform_resume_thread(&procInfo.hThread);

            // Now into the main message loop, listen for hotkeys to send over.
            MessageLoop();
#endif
        }

        vktrace_process_info_delete(&procInfo);
        serverIndex++;
    } while (g_settings.program == NULL);

    vktrace_SettingGroup_delete(&g_settingGroup);
    vktrace_free(g_default_settings.output_trace);

    return 0;
}

