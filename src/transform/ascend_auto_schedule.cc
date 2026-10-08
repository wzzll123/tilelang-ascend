// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*!
 * \file ascend_auto_schedule.cc
 * \brief Register the opt-in A2/A3 scheduling configuration.
 *
 * Task extraction and scheduling live in tilelang/transform/ascend_schedule.
 * Registration is necessary before constructing a native TVM PassContext;
 * a Python Enum entry alone does not register a pass configuration option.
 */

#include <tvm/ir/transform.h>

namespace tvm {
namespace tl {

TVM_REGISTER_PASS_CONFIG_OPTION("tl.ascend_auto_schedule", Bool);

} // namespace tl
} // namespace tvm
