# mkdir hiprt/cache/
echo "// automatically generated, don't edit" > hiprt/cache/Kernels.h
echo "// automatically generated, don't edit" > hiprt/cache/KernelArgs.h

echo "// automatically generated, don't edit" > contrib/Orochi/ParallelPrimitives/cache/Kernels.h
echo "// automatically generated, don't edit" > contrib/Orochi/ParallelPrimitives/cache/KernelArgs.h

python tools/stringify.py ./contrib/Orochi/ParallelPrimitives/RadixSortKernels.h  >> contrib/Orochi/ParallelPrimitives/cache/Kernels.h
python tools/genArgs.py ./contrib/Orochi/ParallelPrimitives/RadixSortKernels.h  >> contrib/Orochi/ParallelPrimitives/cache/KernelArgs.h
python tools/stringify.py ./contrib/Orochi/ParallelPrimitives/RadixSortConfigs.h  >> contrib/Orochi/ParallelPrimitives/cache/Kernels.h

echo "#pragma once" >> hiprt/cache/Kernels.h
echo "#pragma once" >> hiprt/cache/KernelArgs.h

python tools/stringify.py ./hiprt/impl/Math.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/hiprt_vec.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/Aabb.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/AabbList.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/BvhCommon.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/BvhNode.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/Geometry.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/QrDecomposition.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/Quaternion.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/Transform.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/Instance.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/InstanceList.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/MortonCode.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/Scene.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/TriangleMesh.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/Triangle.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/BvhBuilderUtil.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/SbvhCommon.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/ApiNodeList.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/BvhConfig.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/impl/MemoryArena.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/hiprt_types.h 20220318  >> hiprt/cache/Kernels.h
python tools/stringify.py ./hiprt/hiprt_common.h 20220318  >> hiprt/cache/Kernels.h


# hiprt_device_impl.h
python tools/stringify.py ./hiprt/impl/hiprt_device_impl.h 20220318  >> hiprt/cache/Kernels.h
python tools/genArgs.py ./hiprt/impl/hiprt_device_impl.h 20220318  >> hiprt/cache/KernelArgs.h

# hiprt_device.h
python tools/stringify.py ./hiprt/hiprt_device.h 20220318  >> hiprt/cache/Kernels.h
python tools/genArgs.py ./hiprt/hiprt_device.h 20220318  >> hiprt/cache/KernelArgs.h

# BvhBuilderKernels.h
python tools/stringify.py ./hiprt/impl/BvhBuilderKernels.h 20220318  >> hiprt/cache/Kernels.h
python tools/genArgs.py ./hiprt/impl/BvhBuilderKernels.h 20220318  >> hiprt/cache/KernelArgs.h

# LbvhBuilderKernels.h
python tools/stringify.py ./hiprt/impl/LbvhBuilderKernels.h 20220318  >> hiprt/cache/Kernels.h
python tools/genArgs.py ./hiprt/impl/LbvhBuilderKernels.h 20220318  >> hiprt/cache/KernelArgs.h

# PlocBuilderKernels.h
python tools/stringify.py ./hiprt/impl/PlocBuilderKernels.h 20220318  >> hiprt/cache/Kernels.h
python tools/genArgs.py ./hiprt/impl/PlocBuilderKernels.h 20220318  >> hiprt/cache/KernelArgs.h

# SbvhBuilderKernels.h
python tools/stringify.py ./hiprt/impl/SbvhBuilderKernels.h 20220318  >> hiprt/cache/Kernels.h
python tools/genArgs.py ./hiprt/impl/SbvhBuilderKernels.h 20220318  >> hiprt/cache/KernelArgs.h

# BatchBuilderKernels.h
python tools/stringify.py ./hiprt/impl/BatchBuilderKernels.h 20220318  >> hiprt/cache/Kernels.h
python tools/genArgs.py ./hiprt/impl/BatchBuilderKernels.h 20220318  >> hiprt/cache/KernelArgs.h

# BvhImporterKernels.h
python tools/stringify.py ./hiprt/impl/BvhImporterKernels.h 20220318  >> hiprt/cache/Kernels.h
python tools/genArgs.py ./hiprt/impl/BvhImporterKernels.h 20220318  >> hiprt/cache/KernelArgs.h
