#!/usr/bin/env python3
"""
bake_kernels.py — Generate stringified kernel cache headers for HIPRT and Orochi.

Replaces: bakeKernel.bat, bakeKernel.sh, stringify.py, genArgs.py

Outputs (relative to --source-dir):
  hiprt/cache/Kernels.h
  hiprt/cache/KernelArgs.h
  contrib/Orochi/ParallelPrimitives/cache/Kernels.h
  contrib/Orochi/ParallelPrimitives/cache/KernelArgs.h

Usage:
  python tools/bake_kernels.py --source-dir /path/to/hiprt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Stringify: convert a header file into a C string literal
# ---------------------------------------------------------------------------

def stringify(filepath: Path) -> str:
    """Read *filepath* and return a C string literal (multi-line) of its contents.

    - Lines starting with ``//`` (after stripping leading whitespace) are skipped.
    - ``#include`` directives are kept only for the ``hip`` API path.
    - Characters ``\\``, ``"``, ``'`` are escaped.
    """
    lines: list[str] = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\r\n").strip()
            # Skip pure comment lines
            if line.startswith("//"):
                continue
            # Keep #include only for hip
            if "#include" in line:
                # Drop all #include lines (they are resolved at kernel-compile time
                # for hip; only angle-bracket includes of known headers are kept
                # by genArgs separately).
                continue
            # Escape special characters
            escaped = (
                line.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("'", "\\'")
            )
            lines.append(f'"{escaped}\\n"')
    return "\n".join(lines)


def stringify_variable(filepath: Path) -> str:
    """Return a complete ``static const char* varname = ...;`` declaration."""
    stem = filepath.stem  # e.g. "RadixSortKernels"
    varname = f"hip_{stem}"
    body = stringify(filepath)
    return f"static const char* {varname}= \\\n{body}\n;\n"


# ---------------------------------------------------------------------------
# GenArgs: generate kernel include argument arrays
# ---------------------------------------------------------------------------

def gen_args(filepath: Path) -> str:
    """Generate an ``Args[]`` declaration listing angle-bracket includes."""
    stem = filepath.stem
    api = "hip"

    includes_list: list[str] = []  # (varname, original_path)
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if "#include" not in line:
                continue
            # Only process angle-bracket includes: #include <...>
            if "<" not in line or ">" not in line:
                continue
            inner = line.split("<", 1)[1].split(">", 1)[0]
            base = os.path.basename(inner)
            name_stem = base.rsplit(".", 1)[0] if "." in base else base
            varname = f"{api}_{name_stem}"
            includes_list.append((varname, inner))

    out: list[str] = ["#pragma once"]
    out.append(f"namespace {api} {{")
    # Conditional Args array
    out.append(f"#if !defined(HIPRT_LOAD_FROM_STRING) && !defined(HIPRT_BITCODE_LINKING)")
    out.append(f"\tstatic const char** {stem}Args = 0;")
    out.append(f"#else")
    out.append(f"\tstatic const char* {stem}Args[] = {{")
    for varname, _ in includes_list:
        out.append(f"{varname},")
    out.append(f"{api}_{stem}}};")
    out.append(f"#endif")
    # Includes array
    inc_entries = ",".join(f'"{path}"' for _, path in includes_list)
    out.append(f"static const char* {stem}Includes[] = {{{inc_entries}}};")
    out.append(f"}}\t//namespace {api}")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# File manifests (preserving the exact order from the original scripts)
# ---------------------------------------------------------------------------

# Orochi ParallelPrimitives — files relative to source_dir
OROCHI_STRINGIFY_FILES = [
    "contrib/Orochi/ParallelPrimitives/RadixSortKernels.h",
    "contrib/Orochi/ParallelPrimitives/RadixSortConfigs.h",
]
OROCHI_GENARGS_FILES = [
    "contrib/Orochi/ParallelPrimitives/RadixSortKernels.h",
]

# HIPRT — stringify-only files (no genArgs)
HIPRT_STRINGIFY_ONLY_FILES = [
    "hiprt/hiprt_vec.h",
    "hiprt/hiprt_math.h",
    "hiprt/impl/Obb.h",
    "hiprt/impl/Aabb.h",
    "hiprt/impl/AabbList.h",
    "hiprt/impl/BvhCommon.h",
    "hiprt/impl/BvhNode.h",
    "hiprt/impl/Header.h",
    "hiprt/impl/QrDecomposition.h",
    "hiprt/impl/Quaternion.h",
    "hiprt/impl/Transform.h",
    "hiprt/impl/Instance.h",
    "hiprt/impl/InstanceList.h",
    "hiprt/impl/MortonCode.h",
    "hiprt/impl/TriangleMesh.h",
    "hiprt/impl/Triangle.h",
    "hiprt/impl/BvhBuilderUtil.h",
    "hiprt/impl/SbvhCommon.h",
    "hiprt/impl/NodeList.h",
    "hiprt/impl/BvhConfig.h",
    "hiprt/impl/MemoryArena.h",
    "hiprt/hiprt_types.h",
    "hiprt/hiprt_common.h",
]

# HIPRT — files that get both stringify and genArgs
HIPRT_KERNEL_FILES = [
    "hiprt/impl/hiprt_device_impl.h",
    "hiprt/hiprt_device.h",
    "hiprt/impl/BvhBuilderKernels.h",
    "hiprt/impl/LbvhBuilderKernels.h",
    "hiprt/impl/PlocBuilderKernels.h",
    "hiprt/impl/SbvhBuilderKernels.h",
    "hiprt/impl/BatchBuilderKernels.h",
]


# ---------------------------------------------------------------------------
# Atomic file write helper
# ---------------------------------------------------------------------------

def write_file_atomic(path: Path, content: str) -> None:
    """Write *content* to *path*, creating parent dirs if needed.

    Writes to a temporary file first then renames, to avoid partial writes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate stringified kernel cache headers for HIPRT and Orochi."
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        type=Path,
        help="Root of the HIPRT source tree (contains hiprt/, contrib/, etc.)",
    )
    args = parser.parse_args()
    src: Path = args.source_dir.resolve()

    # ------------------------------------------------------------------
    # Orochi ParallelPrimitives
    # ------------------------------------------------------------------
    oro_kernels = "// automatically generated, don't edit\n"
    oro_args = "// automatically generated, don't edit\n"

    for rel in OROCHI_STRINGIFY_FILES:
        oro_kernels += stringify_variable(src / rel)
    for rel in OROCHI_GENARGS_FILES:
        oro_args += gen_args(src / rel)

    write_file_atomic(
        src / "contrib/Orochi/ParallelPrimitives/cache/Kernels.h", oro_kernels
    )
    write_file_atomic(
        src / "contrib/Orochi/ParallelPrimitives/cache/KernelArgs.h", oro_args
    )

    # ------------------------------------------------------------------
    # HIPRT
    # ------------------------------------------------------------------
    hiprt_kernels = "// automatically generated, don't edit\n#pragma once\n"
    hiprt_args = "// automatically generated, don't edit\n#pragma once\n"

    # Stringify-only headers
    for rel in HIPRT_STRINGIFY_ONLY_FILES:
        hiprt_kernels += stringify_variable(src / rel)

    # Kernel headers (stringify + genArgs)
    for rel in HIPRT_KERNEL_FILES:
        hiprt_kernels += stringify_variable(src / rel)
        hiprt_args += gen_args(src / rel)

    write_file_atomic(src / "hiprt/cache/Kernels.h", hiprt_kernels)
    write_file_atomic(src / "hiprt/cache/KernelArgs.h", hiprt_args)

    print(f"bake_kernels.py: generated 4 cache headers in {src}")


if __name__ == "__main__":
    main()
