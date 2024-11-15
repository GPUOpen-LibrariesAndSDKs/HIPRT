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



