import optparse
import os
import sys
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

def compileScript(cmd, dst):
    print('compiling ' + dst + '...')
    sys.stdout.flush()
    return_code = subprocess.call(cmd, shell=True)
    if return_code != 0:
        print(errorMessageHeader + ' executing command: ' + cmd)
    elif not os.path.exists(dst):
        print(errorMessageHeader + ' The file ' + dst + ' does not exist.')
    else:
        print('Compilation SUCCEEDED.')
    sys.stdout.flush()


def compileAmd():
    hipccpath = 'hipcc'
    postfix = '_linux.bc'
    if not isLinux(): # if OS is Windows
        postfix = '_win.bc'
        if which('hipcc', 'bat') == None:
            hipccpath = hipSdkPathFromArgument + '\\bin\\hipcc'
    else: # if OS is Linux
        if hipSdkPathFromArgument != '': # if the hip path it given as an argument
            hipccpath = hipSdkPathFromArgument + '/bin/hipcc'

    hipccpath = common_tools.quoteFilepathIfNeeded(hipccpath)

    cmd = hipccpath + ' --version'
    return_code = subprocess.call(cmd, shell=True)
    if return_code != 0:
        print(errorMessageHeader + ' executing command: ' + cmd)

    result = subprocess.check_output(cmd, shell=True)
    hip_sdk_version = result.decode('utf-8')
    hip_sdk_version_major = re.match(r'HIP version: (\d+).(\d+)', hip_sdk_version).group(1) 
    hip_sdk_version_minor = re.match(r'HIP version: (\d+).(\d+)', hip_sdk_version).group(2)
    hip_sdk_version_num = 10 * int(hip_sdk_version_major) + int(hip_sdk_version_minor)
    hip_version = hip_sdk_version_major +"."+ hip_sdk_version_minor
        
    gpu_archs = common_tools.getAMDGPUArchs(hip_sdk_version_num)

    targets = ''
    for i in gpu_archs:
        targets += ' --offload-arch=' + i

    parallel_jobs = 15

    dst = 'hiprt' + hiprt_ver + '_' + hip_version + '_amd_lib' + postfix
    cmd = hipccpath + ' -x hip ../../hiprt/impl/hiprt_kernels_bitcode.h -O3 -std=c++17 ' + targets + ' -fgpu-rdc -c --gpu-bundle-output -c -emit-llvm -I../../contrib/Orochi/ -I../../ -DHIPRT_BITCODE_LINKING -ffast-math -parallel-jobs=' + str(parallel_jobs) + ' -o ' + dst
    compileScript(cmd, dst)

    dst = 'hiprt' + hiprt_ver + '_' + hip_version + '_amd.hipfb'
    cmd = hipccpath + ' -x hip ../../hiprt/impl/hiprt_kernels.h -O3 -std=c++17 ' + targets + ' -mllvm -amdgpu-early-inline-all=false -mllvm -amdgpu-function-calls=true --genco -I../../ -DHIPRT_BITCODE_LINKING -ffast-math -parallel-jobs=' + str(parallel_jobs) + ' -o ' + dst
    compileScript(cmd, dst)

    dst = 'oro_compiled_kernels.hipfb'
    cmd = hipccpath + ' -x hip ../../contrib/Orochi/ParallelPrimitives/RadixSortKernels.h -O3 -std=c++17 ' + targets + ' --genco -I../../contrib/Orochi/ -include hip/hip_runtime.h -DHIPRT_BITCODE_LINKING -ffast-math -parallel-jobs=' + str(parallel_jobs) + ' -o ' + dst
    compileScript(cmd, dst)


    print('export built programs ...')
    sys.stdout.flush()

    if not os.path.exists(root + 'dist/bin/Debug/'):
        os.makedirs(root + 'dist/bin/Debug/')
    if not os.path.exists(root + 'dist/bin/Release/'):
        os.makedirs(root + 'dist/bin/Release/')

    if isLinux():
        os.system('cp *.hipfb ' + root + 'dist/bin/Debug/')
        os.system('cp *.hipfb ' + root + 'dist/bin/Release/')
        os.system('cp *.bc ' + root + 'dist/bin/Debug/')
        os.system('cp *.bc ' + root + 'dist/bin/Release/')
    else:
        os.system('copy *.hipfb ' + root + 'dist\\bin\\Debug\\')
        os.system('copy *.hipfb ' + root + 'dist\\bin\\Release\\')
        os.system('copy *.bc ' + root + 'dist\\bin\\Debug\\')
        os.system('copy *.bc ' + root + 'dist\\bin\\Release\\')
    sys.stdout.flush()


def compileNv():
    ccbin = ''
    if not isLinux():
        ccbin = common_tools.getVCPath() + '\\bin\\Hostx64\\x64'
        ccbin = '"{}"'.format(ccbin)
        ccbin = '-ccbin=' + ccbin + ' '

    dst = 'hiprt' + hiprt_ver + '_nv_lib.fatbin'
    cmd = 'nvcc -x cu ../../hiprt/impl/hiprt_kernels_bitcode.h -O3 ' + ccbin + '-std=c++17 -fatbin -arch=all --device-c -I../../contrib/Orochi/ -I../../ -DHIPRT_BITCODE_LINKING --use_fast_math --threads 0 -o ' + dst 
    compileScript(cmd, dst)

    dst = 'hiprt' + hiprt_ver + '_nv.fatbin'
    cmd = 'nvcc -x cu ../../hiprt/impl/hiprt_kernels.h -O3 ' + ccbin + '-std=c++17 -fatbin -arch=all -I../../contrib/Orochi/ -I../../ -DHIPRT_BITCODE_LINKING --use_fast_math --threads 0 -o ' + dst
    compileScript(cmd, dst)

    dst = 'oro_compiled_kernels.fatbin'
    cmd = 'nvcc -x cu ../../contrib/Orochi/ParallelPrimitives/RadixSortKernels.h -O3 ' + ccbin + '-std=c++17 -fatbin -arch=all -I../../contrib/Orochi/ -include cuda_runtime.h -DHIPRT_BITCODE_LINKING --use_fast_math --threads 0 -o ' + dst
    compileScript(cmd, dst)

    print('export built programs ...')
    sys.stdout.flush()
    if isLinux():
        os.system('cp *.fatbin ' + root + 'dist/bin/Debug/')
        os.system('cp *.fatbin ' + root + 'dist/bin/Release/')
    else:
        os.system('copy *.fatbin ' + root + 'dist\\bin\\Debug\\')
        os.system('copy *.fatbin ' + root + 'dist\\bin\\Release\\')
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
