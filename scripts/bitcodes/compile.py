import optparse
import os
import sys
import subprocess
import re

def isLinux():
    return os.name != 'nt'


root = '..\\..\\'
if isLinux():
    root = '../../'

errorMessageHeader = 'Bitcodes Compile Error:'


def getVCPath():
    import setuptools.msvc as cc
    pI = cc.PlatformInfo("x64")
    rI = cc.RegistryInfo(pI)
    sI = cc.SystemInfo(rI)
    return sI.VCInstallDir


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
    if not isLinux():
        postfix = '_win.bc'
        if which('hipcc', 'bat') == None:
            hipccpath = root + 'hipSdk\\bin\\hipcc'
            
    cmd = hipccpath + ' --version'
    return_code = subprocess.call(cmd, shell=True)
    if return_code != 0:
        print(errorMessageHeader + ' executing command: ' + cmd)

    result = subprocess.check_output(cmd, shell=True)
    hip_sdk_version = result.decode('utf-8')
    hip_sdk_version_major = re.match(r'HIP version: (\d+).(\d+)', hip_sdk_version).group(1) 
    hip_sdk_version_minor = re.match(r'HIP version: (\d+).(\d+)', hip_sdk_version).group(2)
    hip_version = hip_sdk_version_major +"."+ hip_sdk_version_minor
        
    # llvm.org/docs/AMDGPUUsage.html#processors
    gpus = ['gfx1100', 'gfx1101', 'gfx1102', 'gfx1103',  # Navi3
            'gfx1030', 'gfx1031', 'gfx1032', 'gfx1033', 'gfx1034', 'gfx1035', 'gfx1036',  # Navi2
            'gfx1010', 'gfx1011', 'gfx1012', 'gfx1013',  # Navi1
            'gfx900', 'gfx902', 'gfx904', 'gfx906', 'gfx908', 'gfx909', 'gfx90a', 'gfx90c', 'gfx940', 'gfx941', 'gfx942']  # Vega
    targets = ''
    for i in gpus:
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
        ccbin = getVCPath() + '\\bin\\Hostx64\\x64'
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
parser.add_option('-n', '--nvidia', dest='nv_platform', help='Compile for Nvidia', action='store_true', default=True)
(options, args) = parser.parse_args()

if (options.amd_platform):
    compileAmd()
if (options.nv_platform):
    compileNv()
