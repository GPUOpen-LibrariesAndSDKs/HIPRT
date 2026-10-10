# common_tools.py


import os
import re
import subprocess


def versionKey(version):
    # Sort key that compares dotted version strings numerically by segment, so
    # "14.44.10" > "14.44.9". Non-numeric segments fall back to 0.
    key = []
    for part in version.split("."):
        key.append(int(part) if part.isdigit() else 0)
    return key


def getVCPath(vs_version=None, vc_toolset=None):
    # vs_version: optional vswhere version range selecting the VS *installation*
    #   (e.g. "[17.0,18.0)" for VS2022, "[18.0,19.0)" for VS2026). None => latest.
    # vc_toolset: optional MSVC *toolset* version or prefix (e.g. "14.44",
    #   "14.51"). When given, that toolset is picked from the resolved VS install
    #   instead of the install's default toolset (highest match for a prefix).
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

        # Restrict to a specific VS version range (still picks the newest within it).
        if vs_version:
            vswhere_command += ["-version", vs_version]

        vswhere_output = subprocess.check_output(vswhere_command, encoding='utf-8').strip()

        if not vswhere_output:
            raise IOError("Visual Studio installation not found.")

        msvc_root = os.path.join(vswhere_output, "VC", "Tools", "MSVC")

        if vc_toolset:
            # Select a specific side-by-side MSVC toolset (exact or highest prefix match).
            if not os.path.isdir(msvc_root):
                raise IOError("MSVC toolset directory not found: {}".format(msvc_root))
            candidates = [d for d in os.listdir(msvc_root)
                          if d == vc_toolset or d.startswith(vc_toolset + ".")]
            if not candidates:
                raise IOError("No MSVC toolset matching '{}' under {}. Installed: {}".format(
                    vc_toolset, msvc_root, ", ".join(sorted(os.listdir(msvc_root))) or "<none>"))
            # Pick the newest by numeric version segments (not lexicographically,
            # so "14.44.10" beats "14.44.9").
            vc_tools_version = max(candidates, key=versionKey)
        else:
            vctools_version_file = os.path.join(vswhere_output, "VC", "Auxiliary", "Build", "Microsoft.VCToolsVersion.default.txt")

            if not os.path.exists(vctools_version_file):
                raise IOError("Microsoft.VCToolsVersion.default.txt file not found.")

            with open(vctools_version_file, "r") as file:
                vc_tools_version = file.read().strip()

            if not vc_tools_version:
                raise ValueError("Failed to read VC Tools version.")

        cl_exe_path = os.path.join(msvc_root, vc_tools_version)
        return cl_exe_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError("vswhere command failed: {}".format(e))
    except Exception as e:
        raise RuntimeError("An error occurred: {}".format(e))


def resolveVsVersionRange(vs_version):
    # Map a friendly VS product year to a vswhere version range. If the value is
    # not a known year it is passed through as-is (allowing raw ranges like
    # "[17.0,18.0)" or an exact version). Returns None when nothing is requested.
    if not vs_version:
        return None
    mapping = {
        "2017": "[15.0,16.0)",
        "2019": "[16.0,17.0)",
        "2022": "[17.0,18.0)",
        "2026": "[18.0,19.0)",
    }
    return mapping.get(str(vs_version).strip(), str(vs_version).strip())


# GPU targets HIPRT wants to compile for. Which ones are actually built is
# decided at runtime by getAMDGPUArchs(), which keeps only those supported by
# the HIP SDK in use.
# llvm.org/docs/AMDGPUUsage.html#processors
AMD_GPU_ARCHS = ['gfx1100', 'gfx1101', 'gfx1102', 'gfx1103',  # Navi3
                 'gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035', 'gfx1036',  # Navi2
                 'gfx1010', 'gfx1011', 'gfx1012', 'gfx1013',  # Navi1
                 'gfx900', 'gfx902', 'gfx904', 'gfx906', 'gfx908', 'gfx909', 'gfx90a', 'gfx90c', 'gfx942', 'gfx950',  # Wavefront64 (GFX9/CDNA targets)
                 'gfx1200', 'gfx1201',  # Navi4
                 'gfx1153',  # Krackan
                 'gfx1152', 'gfx1150', 'gfx1151',  # Strix
                 'gfx1170', 'gfx1171', 'gfx115E'   # Medusa
                ]


def findHipClangPath(hip_sdk_path=None, hipcc_version_output=None):
    # Locate the clang shipped with the HIP SDK. Only that clang is accepted as
    # an authority on supported GPU targets: one picked up from the PATH could
    # belong to a different toolchain than the one used for the build.
    # Returns None when it cannot be found.
    candidate_dirs = []

    # 'hipcc --version' prints clang's own InstalledDir, so this follows
    # whichever hipcc is actually used, whatever the SDK layout. Callers already
    # run that command to read the SDK version, so it costs nothing.
    if hipcc_version_output:
        match = re.search(r'^InstalledDir:\s*(.+?)\s*$', hipcc_version_output, re.MULTILINE)
        if match:
            candidate_dirs.append(match.group(1))

    # Fallback on the usual layouts, relative to the SDK root: Windows keeps
    # clang in bin/, Linux in lib/llvm/bin/ (with llvm/ as a symlink).
    if hip_sdk_path:
        for sub_dirs in [('bin',), ('lib', 'llvm', 'bin'), ('llvm', 'bin')]:
            candidate_dirs.append(os.path.join(hip_sdk_path, *sub_dirs))

    exe_suffix = '.exe' if os.name == 'nt' else ''
    for directory in candidate_dirs:
        # Some distributions only ship the 'amdclang' alias next to hipcc.
        for exe_name in ['clang', 'amdclang']:
            clang_path = os.path.join(directory, exe_name + exe_suffix)
            if os.path.isfile(clang_path):
                return clang_path

    return None


def queryClangSupportedArchs(clang_path):
    # Ask the AMDGPU backend which processors it can generate code for. Returns
    # None if the query fails. '-nogpulib' is required: without it the driver
    # stops on a missing device library before printing anything. No input file
    # is needed.
    cmd = [clang_path, '--target=amdgcn-amd-amdhsa', '-nogpulib', '-mcpu=help']
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
    except subprocess.CalledProcessError as e:
        # Some clang versions report the list as a diagnostic and exit non-zero.
        output = e.output or ''
    except OSError:
        return None

    # Recent clang prints one target per line, older ones inline them ("valid
    # target CPU values are: gfx900, gfx902, ..."); splitting on whitespace and
    # commas handles both. Generic targets such as 'gfx10-3-generic' do not
    # match the pattern and are ignored.
    archs = set(t for t in re.split(r'[\s,]+', output) if re.match(r'^gfx[0-9a-zA-Z]+$', t))

    return archs or None


def findDeviceLibDir(hip_sdk_path=None, clang_path=None):
    # ROCm device libraries live in <root>/amdgcn/bitcode. Look for that folder
    # from the SDK root and by walking up from clang, since the depth differs
    # between OSes and SDK flavours. Returns None when not found, in which case
    # the device library check is simply skipped.
    roots = []

    if hip_sdk_path:
        roots.append(hip_sdk_path)

    if clang_path:
        directory = os.path.dirname(clang_path)
        for _ in range(4):
            directory = os.path.dirname(directory)
            roots.append(directory)

    for root in roots:
        bitcode_dir = os.path.join(root, 'amdgcn', 'bitcode')
        if os.path.isdir(bitcode_dir):
            return bitcode_dir

    return None


def getAMDGPUArchs(hip_sdk_path=None, hipcc_version_output=None):
    # Return the GPU targets to compile for: the targets of AMD_GPU_ARCHS that
    # the HIP SDK in use really supports, logging the verdict for each of them.
    # Not being able to ask the SDK is an error: compiling for a target it does
    # not support fails anyway, only later and with a more obscure message.
    clang_path = findHipClangPath(hip_sdk_path, hipcc_version_output)
    if not clang_path:
        raise RuntimeError('Cannot find the clang shipped with the HIP SDK, needed to know '
                           'which GPU targets that SDK supports.')

    supported_archs = queryClangSupportedArchs(clang_path)
    if not supported_archs:
        raise RuntimeError('Cannot get the list of GPU targets supported by ' + clang_path)

    device_lib_dir = findDeviceLibDir(hip_sdk_path, clang_path)

    print('AMD GPU targets (supported list from {}):'.format(clang_path))

    gpus_archs = []
    for arch in AMD_GPU_ARCHS:
        enabled = arch in supported_archs
        reason = 'not supported by this toolchain'

        # A target the compiler knows about but with no device library would
        # only fail later, when compiling the kernels; drop it here instead.
        if enabled and device_lib_dir:
            isa_lib = 'oclc_isa_version_' + arch[len('gfx'):] + '.bc'
            if not os.path.exists(os.path.join(device_lib_dir, isa_lib)):
                enabled = False
                reason = 'no device library (' + isa_lib + ')'

        line = '  [{}] {:<9}'.format('OK' if enabled else '--', arch)
        if not enabled:
            line += reason
        print(line.rstrip())

        if enabled:
            gpus_archs.append(arch)

    if not gpus_archs:
        raise RuntimeError('None of the HIPRT GPU targets is supported by ' + clang_path)

    return gpus_archs


# encapsulate the full path in quotes if it's not already quoted
def quoteFilepathIfNeeded(path: str) -> str:
    if not (path.startswith('"') and path.endswith('"')):
        path = '"' + path + '"'
    return path


def remove_trailing_slash(path):
    if path.endswith('/') or path.endswith('\\'):
        return path[:-1]
    return path
