# common_tools.py


import os
import subprocess


def getVCPath():
    try:
        vswhere_default_path = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
        
        if not os.path.exists(vswhere_default_path):
            raise IOError("vswhere not found at the default location: {}".format(vswhere_default_path))

        vswhere_command = [
            vswhere_default_path,
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath"
        ]

        vswhere_output = subprocess.check_output(vswhere_command, encoding='utf-8').strip()

        if not vswhere_output:
            raise IOError("Visual Studio installation not found.")

        vctools_version_file = os.path.join(vswhere_output, "VC", "Auxiliary", "Build", "Microsoft.VCToolsVersion.default.txt")

        if not os.path.exists(vctools_version_file):
            raise IOError("Microsoft.VCToolsVersion.default.txt file not found.")

        with open(vctools_version_file, "r") as file:
            vc_tools_version = file.read().strip()

        if not vc_tools_version:
            raise ValueError("Failed to read VC Tools version.")

        cl_exe_path = os.path.join(vswhere_output, "VC", "Tools", "MSVC", vc_tools_version)
        return cl_exe_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError("vswhere command failed: {}".format(e))
    except Exception as e:
        raise RuntimeError("An error occurred: {}".format(e))


def getAMDGPUArchs(hip_sdk_version_num):
    # llvm.org/docs/AMDGPUUsage.html#processors
    gpus_archs = ['gfx1100', 'gfx1101', 'gfx1102', 'gfx1103',  # Navi3
                  'gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035', 'gfx1036',  # Navi2
                  'gfx1010', 'gfx1011', 'gfx1012', 'gfx1013',  # Navi1
                  'gfx900', 'gfx902', 'gfx904', 'gfx906', 'gfx908', 'gfx909', 'gfx90a', 'gfx90c', 'gfx942']  # Vega
    

    if hip_sdk_version_num >= 63:
        gpus_archs.append('gfx1152')

    if hip_sdk_version_num >= 62: # Navi4 supported from 6.2
        gpus_archs.append('gfx1200')
        gpus_archs.append('gfx1201')

    if hip_sdk_version_num >= 61: # Strix supported from 6.1
        gpus_archs.append('gfx1150')
        gpus_archs.append('gfx1151')

    return gpus_archs


# encapsulate the full path in quotes if it contains spaces and is not already quoted
def quoteFilepathIfNeeded(path: str) -> str:
    if " " in path and not (path.startswith('"') and path.endswith('"')):
        path = '"' + path + '"'
    return path

