workspace "ee"
   configurations { "Debug", "Release" }
   language "C++"
   platforms "x64"
   architecture "x86_64"

   project "ee"
      kind "ConsoleApp"

      targetdir "dist/bin/%{cfg.buildcfg}"
      location "build/"

		sysincludedirs { "./" }

		files { "cl.cpp" }

      filter {"platforms:x64", "configurations:Debug"}
         targetsuffix "64D"
         defines { "DEBUG" }
         symbols "On"

      filter {"platforms:x64", "configurations:Release"}
         targetsuffix "64"
         defines { "NDEBUG" }
         optimize "On"
