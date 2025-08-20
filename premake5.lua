include "./tools/functions.lua"

newoption {
    trigger = "bakeKernel",
    description = "Encrypt and bake kernels"
}

newoption {
    trigger = "bitcode",
    description = "Enable bitcode linking"
}

newoption {
    trigger = "precompile",
    description = "Precompile kernels"
}

newoption {
    trigger = "hiprtew",
    description = "Use hiprtew"
}

newoption {
    trigger = "noUnittest",
    description = "Don't build unit tests",
}

newoption {
    trigger = "noEncrypt",
    description = "Don't encrypt kernel source and binaries",
}

hip_sdk_version, hip_final_path = get_hip_sdk_verion()
print( "HIP_VERSION_STR: "..hip_sdk_version )
if hip_final_path ~= nil then
	print( "HIP SDK path: " .. hip_final_path )
else
	print( "no HIP SDK folder found." )
end

workspace "hiprt"
    configurations {"Debug", "Release", "RelWithDebInfo", "DebugGpu" }
    language "C++"
    platforms "x64"
    architecture "x86_64"

	if os.ishost("windows") then
		defines {"__WINDOWS__"}
	end
    characterset("MBCS")

    filter {"platforms:x64", "configurations:Debug or configurations:DebugGpu"}
      targetsuffix "64D"
      defines {"DEBUG"}
      symbols "On"
    filter {"platforms:x64", "configurations:DebugGpu"}
      defines {"DEBUG_GPU"}
    filter {"platforms:x64", "configurations:Release or configurations:RelWithDebInfo"}
      targetsuffix "64"
      defines {"NDEBUG"}
      optimize "On"
    filter {"platforms:x64", "configurations:RelWithDebInfo"}
      symbols "On"
    filter {}
	flags { "MultiProcessorCompile" }

    if os.ishost("windows") then
        buildoptions {"/wd4244", "/wd4305", "/wd4018", "/wd4996", "/Zc:__cplusplus"}
    end
    if os.ishost("linux") then
        buildoptions {"-fvisibility=hidden"}
    end
    defines {"__USE_HIP__"}

    -- this define is to identify that we are on the public repository of HIPRT.
    -- it helps AMD to maintain both a public and a private repo for experimentation.
    defines {"HIPRT_PUBLIC_REPO"}
    
    

    -- enable CUDA if possible
    include "./contrib/Orochi/Orochi/enable_cuew"

    targetdir "dist/bin/%{cfg.buildcfg}"    
    location "build/"
    
    write_version_info("./hiprt/hiprt.h.in", "./hiprt/hiprt.h", "version.txt", hip_sdk_version)
	write_version_info("./hiprt/hiprtew.h.in", "./hiprt/hiprtew.h", "version.txt", hip_sdk_version)

	if not _OPTIONS["noUnittest"] then
		startproject "unittest"
	end

    HIPRT_NAME = get_hiprt_library_name("version.txt")
    project( HIPRT_NAME )
        cppdialect "C++17"
        kind "SharedLib"
        defines {"HIPRT_EXPORTS"}
	if _OPTIONS["bitcode"] then
		defines {"HIPRT_BITCODE_LINKING"}
        defines {"ORO_PRECOMPILED"}
	end

    if not _OPTIONS["noEncrypt"] then
        defines {"HIPRT_ENCRYPT"}
    end

    if _OPTIONS["precompile"] then
		cmdExec = "cd ./scripts/bitcodes/ && python compile.py"
		if hip_final_path ~= nil then
			cmdExec = cmdExec .. " --hipSdkPath \"" .. hip_final_path .. "\""
		end
		print("Executing: " .. cmdExec);
        os.execute( cmdExec )
    end

    if _OPTIONS["bakeKernel"] or _OPTIONS["bitcode"] then
        print(">> BakeKernel Executed")
        if os.ishost("windows") then
            os.execute("mkdir hiprt\\cache")
            os.execute("tools\\bakeKernel.bat")
        else
            os.execute("mkdir hiprt/cache")
            os.execute("./tools/bakeKernel.sh")
        end
        if _OPTIONS["bakeKernel"] then
            defines {"HIPRT_LOAD_FROM_STRING"}
            defines { "ORO_PP_LOAD_FROM_STRING" }
        end
        defines {"HIPRT_BAKE_KERNEL_GENERATED"}
    end
    if os.istarget("windows") then
        links{ "version" }
    end

    externalincludedirs {"./"}
    files {"hiprt/**.h", "hiprt/**.cpp"}
    externalincludedirs { "./contrib/Orochi/" }
    files {"contrib/Orochi/Orochi/**.h", "contrib/Orochi/Orochi/**.cpp"}
    files {"contrib/Orochi/contrib/cuew/**.h", "contrib/Orochi/contrib/cuew/**.cpp"}
    files {"contrib/Orochi/contrib/hipew/**.h", "contrib/Orochi/contrib/hipew/**.cpp"}
    files {"contrib/Orochi/ParallelPrimitives/**.h", "contrib/Orochi/ParallelPrimitives/**.cpp"}

	if not _OPTIONS["noUnittest"] then
		project( "unittest" )
			cppdialect "C++17"
			kind "ConsoleApp"
			if _OPTIONS["bitcode"] then
				defines {"HIPRT_BITCODE_LINKING"}
			end
			if os.ishost("windows") then
				buildoptions { "/wd4244" }
				links{ "version" }
			end
			externalincludedirs {"./"}
			links { HIPRT_NAME }
			
			if os.ishost("linux") then
				links { "pthread", "dl" }
			end
			files { "test/hiprtT*.h", "test/hiprtT*.cpp", "test/shared.h", "test/main.cpp", "test/CornellBox.h", "test/kernels/*.h" }
			externalincludedirs { "./contrib/Orochi/" }
			files {"contrib/Orochi/Orochi/**.h", "contrib/Orochi/Orochi/**.cpp"}
			files {"contrib/Orochi/contrib/cuew/**.h", "contrib/Orochi/contrib/cuew/**.cpp"}
			files {"contrib/Orochi/contrib/hipew/**.h", "contrib/Orochi/contrib/hipew/**.cpp"}

			files { "contrib/gtest-1.6.0/gtest-all.cc" }
			externalincludedirs { "contrib/gtest-1.6.0/" }
			defines { "GTEST_HAS_TR1_TUPLE=0" }
			externalincludedirs { "contrib/embree/include/" }
			if os.istarget("windows") then
				libdirs{"contrib/embree/win/"}
				copydir( "./contrib/embree/win", "./dist/bin/Release/", "*.dll" )
				copydir( "./contrib/embree/win", "./dist/bin/Debug/", "*.dll" )
				libdirs{"contrib/bin/win64"}
				copydir( "./contrib/Orochi/contrib/bin/win64", "./dist/bin/Release/", "*.dll" )
				copydir( "./contrib/Orochi/contrib/bin/win64", "./dist/bin/Debug/", "*.dll" )
			end
			if os.istarget("linux") then
				libdirs{"contrib/embree/linux/"}
			end
			links{ "embree4", "tbb" }

	end

	if _OPTIONS["hiprtew"] then
		 project( "hiprtewtest" )
                 cppdialect "C++17"
				 kind "ConsoleApp"
				 defines {"HIPRT_EXPORTS"}
				 defines {"USE_HIPRTEW"}
				 if os.ishost("windows") then
						 buildoptions { "/wd4244" }
						 links{ "version" }
				 end
				 externalincludedirs {"./", "./contrib/Orochi/"}
				 if os.ishost("linux") then
						 links { "pthread", "dl"}
				 end
				 files {"contrib/Orochi/Orochi/**.h", "contrib/Orochi/Orochi/**.cpp"}
				 files {"contrib/Orochi/contrib/**.h", "contrib/Orochi/contrib/**.cpp"}
				 files { "test/hiprtewTest.h", "test/hiprtewTest.cpp" }

				 files { "contrib/gtest-1.6.0/gtest-all.cc" }
				 externalincludedirs { "contrib/gtest-1.6.0/" }
				 defines { "GTEST_HAS_TR1_TUPLE=0" }
    end
