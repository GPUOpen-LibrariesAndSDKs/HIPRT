import optparse
import os
import sys
import shutil
from sys import prefix
import subprocess
import re
import common_tools

def isLinux():
    return os.name != 'nt'


root = '..\\..\\'
if isLinux():
    root = '../../'

errorMessageHeader = 'Bitcodes Compile Error:'

hipSdkPathFromArgument = ''

def getVersion():
    f = open(root + 'version.txt', 'r')
    HIPRT_MAJOR_VERSION = int(f.readline())
    HIPRT_MINOR_VERSION = int(f.readline())
    HIPRT_VERSION = HIPRT_MAJOR_VERSION * 1000 + HIPRT_MINOR_VERSION
    return '%05d' % HIPRT_VERSION


hiprt_ver = getVersion()

def remove_trailing_slash(path):
    if path.endswith('/') or path.endswith('\\'):
        return path[:-1]
    return path

def which(program, sufix=''):
    sufix = '.' + sufix
    if isLinux():
        sufix = ''

    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program + sufix):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program + sufix)
            if is_exe(exe_file):
                return exe_file
    return None


def compileScript(msg, cmd, dst):
    print(msg + dst + '...')
    sys.stdout.flush()
    return_code = subprocess.call(cmd, shell=True)
    if return_code != 0:
        print(errorMessageHeader + ' executing command: ' + cmd)
        sys.exit(1)
    elif not os.path.exists(dst):
        print(errorMessageHeader + ' The file ' + dst + ' does not exist.')
        sys.exit(1)
    else:
        print('Compilation SUCCEEDED.')
    sys.stdout.flush()


def compileAmd():
    hipccpath = 'hipcc'
    postfix = '_linux.bc'
    postfixLink = '_linux.hipfb'
    hiprtlibpath = root + 'dist/bin/Release/'
    if not isLinux(): # if OS is Windows
        clangpath = 'clang++'
        postfix = '_win.bc'
        postfixLink = '_win.hipfb'
        hiprtlibpath = root + 'dist\\bin\\Release\\'
        if which('hipcc', 'bat') == None:
            hipccpath = hipSdkPathFromArgument + '\\bin\\hipcc'
        if which('clang++', 'exe') == None:
            clangpath = hipSdkPathFromArgument + '\\bin\\clang++'
    else: # if OS is Linux
        if hipSdkPathFromArgument != '': # if the hip path it given as an argument
            hipccpath = hipSdkPathFromArgument + '/bin/hipcc'
            clangpath = hipSdkPathFromArgument + '/bin/clang++'
            if not os.path.exists(clangpath):
                clangpath = hipSdkPathFromArgument + '/bin/amdclang++'
        else:
            clangpath = '/opt/rocm/bin/amdclang++'

    hipccpath = common_tools.quoteFilepathIfNeeded(hipccpath)
    clangpath = common_tools.quoteFilepathIfNeeded(clangpath)

    result = subprocess.check_output(hipccpath + ' --version', shell=True)
    hip_sdk_version = result.decode('utf-8')
    hip_sdk_version_major = re.match(r'HIP version: (\d+).(\d+)', hip_sdk_version).group(1) 
    hip_sdk_version_minor = re.match(r'HIP version: (\d+).(\d+)', hip_sdk_version).group(2)
    hip_sdk_version_num = 10 * int(hip_sdk_version_major) + int(hip_sdk_version_minor)
    hip_version = hip_sdk_version_major +"."+ hip_sdk_version_minor
    
    gpu_archs = common_tools.getAMDGPUArchs(hip_sdk_version_num)

    targets = ''
    for i in gpu_archs:
        targets += ' --offload-arch=' + i

    # compile hiprt traversal code
    hiprt_lib = hiprtlibpath + 'hiprt' + hiprt_ver + '_' + hip_version + '_amd_lib' + postfix
    

    # compile custom function table
    hiprt_custom_func = 'hiprt' + hiprt_ver + '_' + hip_version + '_custom_func_table.bc'
    cmd = hipccpath + ' -O3 -std=c++17 ' + targets + ' -fgpu-rdc -c --gpu-bundle-output -c -emit-llvm -I../../ -ffast-math ../../test/bitcodes/custom_func_table.cpp -parallel-jobs=15 -o ' + hiprt_custom_func
    compileScript('compiling ', cmd, hiprt_custom_func)

    # compiling unit test
    hiprt_unit_test = 'hiprt' + hiprt_ver + '_' + hip_version + '_unit_test'+ postfix
    cmd = hipccpath + ' -O3 -std=c++17 ' + targets + ' -fgpu-rdc -c --gpu-bundle-output -c -emit-llvm -I../../ -ffast-math -D BLOCK_SIZE=64 -D SHARED_STACK_SIZE=16 ../../test/bitcodes/unit_test.cpp -parallel-jobs=15 -o ' + hiprt_unit_test
    compileScript('compiling ', cmd, hiprt_unit_test)

    # linking
    offline_unit_test_linked = 'hiprt' + hiprt_ver + '_' + hip_version + '_precompiled_bitcode' + postfixLink
    cmd = clangpath + ' -fgpu-rdc --hip-link --cuda-device-only ' + targets + ' ' + hiprt_custom_func + ' ' + hiprt_unit_test + ' ' + hiprt_lib + ' -o ' + offline_unit_test_linked
    compileScript('linking ', cmd, offline_unit_test_linked)


    print('export built programs ...')
    sys.stdout.flush()

    if not os.path.exists(root + 'dist/bin/Debug/'):
        os.makedirs(root + 'dist/bin/Debug/')
    if not os.path.exists(root + 'dist/bin/Release/'):
        os.makedirs(root + 'dist/bin/Release/')
    if not os.path.exists(root + 'hiprt/bitcodes/'):
        os.makedirs(root + 'hiprt/bitcodes/')

    if isLinux():
        os.system('cp *.hipfb ' + root + 'dist/bin/Debug/')
        os.system('cp *.hipfb ' + root + 'dist/bin/Release/')
        os.system('cp *.hipfb ' + root + 'hiprt/bitcodes/')
    else:
        os.system('copy *.hipfb ' + root + 'dist\\bin\\Debug\\')
        os.system('copy *.hipfb ' + root + 'dist\\bin\\Release\\')
        os.system('copy *.hipfb ' + root + 'hiprt\\bitcodes\\')
    sys.stdout.flush()

    print('clean temp files ...')
    sys.stdout.flush()
    os.remove(hiprt_custom_func)
    os.remove(hiprt_unit_test)
    sys.stdout.flush()

def compileNv():
    ccbin = ''
    if not isLinux():
        ccbin = common_tools.getVCPath() + '\\bin\\Hostx64\\x64'
        ccbin = '"{}"'.format(ccbin)
        ccbin = '-ccbin=' + ccbin

    # TODO: implement nvidia
    print('TODO: CURRENTLY compileNv IS NOT IMPLEMENTED.');


    print('export built programs ...')
    sys.stdout.flush()
    if isLinux():
        os.system('cp *.fatbin ' + root + 'dist/bin/Debug/')
        os.system('cp *.fatbin ' + root + 'dist/bin/Release/')
        os.system('cp *.fatbin ' + root + 'hiprt/bitcodes/')
    else:
        os.system('copy *.fatbin ' + root + 'dist\\bin\\Debug\\')
        os.system('copy *.fatbin ' + root + 'dist\\bin\\Release\\')
        os.system('copy *.fatbin ' + root + 'hiprt\\bitcodes\\')
    sys.stdout.flush()


parser = optparse.OptionParser()
parser.add_option('-a', '--amd', dest='amd_platform', help='Compile for AMD', action='store_true', default=True)
parser.add_option('--no-amd'   , dest='amd_platform', help='Do not compile for AMD', action='store_false' )
parser.add_option('-n', '--nvidia', dest='nv_platform', help='Compile for Nvidia', action='store_true', default=False)

# Add the optional hipSdkPath argument
parser.add_option(
    "-p", "--hipSdkPath",
    dest="hipSdkPath",
    default=None,
    help="Path to the HIP SDK",
    metavar="PATH"
    )

(options, args) = parser.parse_args()


if options.hipSdkPath:
    hipSdkPathFromArgument = remove_trailing_slash(options.hipSdkPath)
    print('Compile kernel using hip sdk: ' + hipSdkPathFromArgument)
else:
    if not isLinux():
        hipSdkPathFromArgument = root + 'hipSdk\\'


if (options.amd_platform):
    compileAmd()
if (options.nv_platform):
    compileNv()
