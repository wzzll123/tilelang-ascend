// Copyright (c) Tile-AI Corporation.
// Licensed under the MIT License.

/*! \file ascend_vector_semantic_projection.cc
 * \brief Validated functional/synchronization view of selected Vector IR.
 */

#include <tvm/node/structural_equal.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../op/ascend.h"
#include "common/ascend_vector_mask.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

tvm::transform::Pass AscendVectorInstructionSelection(Target, std::string);
tvm::transform::Pass AscendVectorMaskLegalize(Target, std::string);

namespace {

// Do not independently reproduce the terminal catalog or mask dataflow.  The
// inverse below is accepted only when the existing selector and legalizer can
// reconstruct the original body exactly, including its mask-state repairs.
class SemanticProjector final : public StmtExprMutator {
public:
  int selected_count{0};
  int setter_count{0};

private:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != "resource_scope") {
      return StmtExprMutator::VisitStmt_(op);
    }
    const auto *scope = op->value.as<IntImmNode>();
    ICHECK(scope && (scope->value == 0 || scope->value == 1));
    int previous = scope_;
    scope_ = static_cast<int>(scope->value);
    Stmt body = VisitStmt(op->body);
    scope_ = previous;
    return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> body;
    for (const Stmt &stmt : op->seq) {
      Stmt projected = VisitStmt(stmt);
      if (!is_no_op(projected)) {
        body.push_back(projected);
      }
    }
    return body.empty() ? Evaluate(0) : SeqStmt::Flatten(body);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *node = op->value.as<CallNode>();
    if (node == nullptr) {
      return StmtExprMutator::VisitStmt_(op);
    }
    Call call = GetRef<Call>(node);
    if (IsVectorMaskSetter(call)) {
      ++setter_count;
      ICHECK_EQ(scope_, 1) << "Vector-mask setter outside Vector scope";
      if (call->op.same_as(ascend_set_mask_mode())) {
        ICHECK_EQ(call->args.size(), 1U);
        const auto *mode = call->args[0].as<IntImmNode>();
        ICHECK(mode && (mode->value == 0 || mode->value == 1));
      } else {
        ICHECK_EQ(call->args.size(), 2U);
        for (const PrimExpr &word : call->args) {
          // NORMAL terminals carry uint64 bit masks, but COUNTER repairs
          // preserve the selected count's integer dtype (commonly int32).
          // Repair() and MaskSetterCodegen() intentionally accept both.
          DataType dtype = word.dtype();
          ICHECK((dtype.is_int() || dtype.is_uint()) && dtype.lanes() == 1)
              << "Vector-mask payload must be a scalar integer";
        }
      }
      return Evaluate(0);
    }
    if (IsSelectedVectorTerminal(call)) {
      ICHECK_EQ(scope_, 1) << "Selected Vector terminal outside Vector scope";
      SelectedCallView view(call);
      ++selected_count;
      return Evaluate(Call(call->dtype, view.semantic_spec().base,
                           view.semantic_args(), call->span),
                      op->span);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Call call = GetRef<Call>(op);
    ICHECK(!IsSelectedVectorTerminal(call) && !IsVectorMaskSetter(call))
        << "Selected Vector calls and mask setters must be top-level Evaluate";
    return StmtExprMutator::VisitExpr_(op);
  }

  int scope_{-1};
};

} // namespace

tvm::transform::Pass AscendVectorSemanticProjection(Target target,
                                                    std::string platform) {
  auto pass_func = [=](PrimFunc func, IRModule, PassContext) {
    if (!UseCompilerManagedVectorMask(target, platform)) {
      return func;
    }
    SemanticProjector projector;
    Stmt body = projector(func->body);
    if (projector.selected_count == 0 && projector.setter_count == 0) {
      return func;
    }
    PrimFunc semantic = func;
    semantic.CopyOnWrite()->body = body;
    IRModule round_trip = IRModule::FromExpr(semantic);
    round_trip = AscendVectorInstructionSelection(target, platform)(round_trip);
    round_trip = AscendVectorMaskLegalize(target, platform)(round_trip);
    PrimFunc rebuilt = Downcast<PrimFunc>(round_trip->Lookup("main"));
    ICHECK(StructuralEqual()(func->body, rebuilt->body))
        << "Vector semantic projection requires canonical, fully legalized "
           "IR: terminal payload or mask-state repairs differ on round trip";
    // This view is not a cycle-cost preserving lowering.  Its consumers must
    // not issue a performance report that silently omits physical mask work.
    return WithAttr(std::move(semantic),
                    "tilelang.simulator_vector_semantic_projection",
                    Integer(1));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.AscendVectorSemanticProjection",
                            {});
}

TVM_REGISTER_GLOBAL("tl.transform.AscendVectorSemanticProjection")
    .set_body_typed(AscendVectorSemanticProjection);

} // namespace tl
} // namespace tvm
