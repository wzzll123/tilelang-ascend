# Copyright (c) Tile-AI Corporation.
# Licensed under the MIT License.
"""Execute the native helper with DMA stubs; do not assume zero-block DMA is legal."""

from pathlib import Path
import shutil
import subprocess

import pytest


def test_ub_to_gm_empty_rectangle_never_issues_dma(tmp_path):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("A host C++ compiler is required for the native helper test")
    header = Path(__file__).resolve().parents[3] / "src/tl_templates/ascend/common.h"
    text = header.read_text()
    start = text.index("template <typename T, uint32_t srcN, uint32_t srcM = 1>\n"
                       "CATLASS_DEVICE void\ncopy_ub_to_gm(")
    end = text.index("template <typename T, uint32_t srcN, uint32_t srcM = 1>", start + 1)
    helper = text[start:end]
    source = tmp_path / "empty_dma.cc"
    source.write_text("""
using uint32_t = unsigned int;
#define CATLASS_DEVICE
template <typename T> struct GlobalTensor {};
template <typename T> struct LocalTensor {};
namespace AscendC {
int calls = 0;
struct DataCopyExtParams {
  DataCopyExtParams(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t) {}
};
template <typename T>
void DataCopyPad(GlobalTensor<T>, LocalTensor<T>, DataCopyExtParams) { ++calls; }
}
""" + helper + """
int main() {
  GlobalTensor<float> dst;
  LocalTensor<float> src;
  copy_ub_to_gm<float, 128, 16>(dst, src, 128, 0, 128);
  copy_ub_to_gm<float, 128, 16>(dst, src, 128, 16, 0);
  if (AscendC::calls != 0) return 1;
  copy_ub_to_gm<float, 128, 16>(dst, src, 128, 1, 128);
  copy_ub_to_gm<float, 128, 16>(dst, src, 128, 15, 128);
  copy_ub_to_gm<float, 128, 16>(dst, src, 128, 16, 128);
  if (AscendC::calls != 3) return 2;
}
""")
    executable = tmp_path / "empty_dma"
    subprocess.run([compiler, "-std=c++17", str(source), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
