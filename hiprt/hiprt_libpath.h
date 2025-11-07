//////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.
//
//////////////////////////////////////////////////////////////////////////////////////////

#pragma once

//
// Define the library path that HIPRT will use.
// Order matters: the first library file of the list to exist will be the one loaded.
//

#ifdef _WIN32

const char* g_hip_paths[] = {
	"amdhip64_7.dll",
	"amdhip64_6.dll",
	"amdhip64.dll", // <- hip '5.x' DLL.
	NULL };

const char* g_hiprtc_paths[] = { "hiprtc0707.dll", "hiprtc0706.dll", "hiprtc0705.dll", "hiprtc0704.dll", "hiprtc0703.dll",
								 "hiprtc0702.dll", "hiprtc0701.dll", "hiprtc0700.dll", "hiprtc0605.dll", "hiprtc0604.dll",
								 "hiprtc0603.dll", "hiprtc0602.dll", "hiprtc0601.dll", "hiprtc0600.dll", "hiprtc0507.dll",
								 "hiprtc0506.dll", "hiprtc0505.dll", "hiprtc0504.dll", "hiprtc0503.dll", NULL };
#elif defined( __APPLE__ )

const char** g_hip_paths	= nullptr;
const char** g_hiprtc_paths = nullptr;
#else

const char* g_hip_paths[] = {

	// first, we try with the generic symbolic link.
	"libamdhip64.so",
	"/opt/rocm/lib/libamdhip64.so",
	"/opt/rocm/hip/lib/libamdhip64.so",

	// .. if it doesn't exist, we try the specific versions
	"libamdhip64.so.7",
	"/opt/rocm/lib/libamdhip64.so.7",
	"/opt/rocm/hip/lib/libamdhip64.so.7",

	"libamdhip64.so.6",
	"/opt/rocm/lib/libamdhip64.so.6",
	"/opt/rocm/hip/lib/libamdhip64.so.6",

	"libamdhip64.so.5",
	"/opt/rocm/lib/libamdhip64.so.5",
	"/opt/rocm/hip/lib/libamdhip64.so.5",

	NULL };

const char* g_hiprtc_paths[] = {

	// first, we try with the generic symbolic link.
	"/opt/rocm/hip/lib/libhiprtc.so",
	"/opt/rocm/lib/libhiprtc.so",
	"libhiprtc.so",

	// .. if it doesn't exist, we try the specific versions
	"/opt/rocm/hip/lib/libhiprtc.so.7",
	"/opt/rocm/lib/libhiprtc.so.7",
	"libhiprtc.so.7",

	"/opt/rocm/hip/lib/libhiprtc.so.6",
	"/opt/rocm/lib/libhiprtc.so.6",
	"libhiprtc.so.6",

	"/opt/rocm/hip/lib/libhiprtc.so.5",
	"/opt/rocm/lib/libhiprtc.so.5",
	"libhiprtc.so.5",

	NULL };
#endif
