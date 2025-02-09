#include "operators/matmul.h"

namespace infini {

MatmulObj::MatmulObj(GraphObj *graph, Tensor A, Tensor B, Tensor C, bool transA, bool transB)
    : OperatorObj(OpType::MatMul, TensorVec{A, B}, {C}), transA(transA), transB(transB) {
  IT_ASSERT(checkValid(graph));
}

string MatmulObj::toString() const {
  std::ostringstream os;
  os << "Matmul([" << (transA ? "A^T" : "A") << "," << (transB ? "B^T" : "B]") << ",A=" << inputs[0]->getGuid()
     << ",B=" << inputs[1]->getGuid() << ",C=" << outputs[0]->getGuid() << ",mnk=[" << m << "," << n << "," << k
     << "])";
  return os.str();
}

optional<vector<Shape>> MatmulObj::inferShape(const TensorVec &inputs) {
  // =================================== 作业 ===================================
  // TODO：返回经过 matmul 操作后的 shape
  // REF: https://github.com/onnx/onnx/blob/main/docs/Operators.md#gemm
  // =================================== 作业 ===================================

  // 分别保存第一个和第二个输入矩阵的宽高
  int m1, n1, m2, n2;

  // 获取输入张量的维度
  int inputsShapeDim = inputs[0]->getDims().size();
  // 获取最后两个维度作为矩阵的宽高
  m1 = inputs[0]->getDims()[inputsShapeDim - 2];
  n1 = inputs[0]->getDims()[inputsShapeDim - 1];
  m2 = inputs[1]->getDims()[inputsShapeDim - 2];
  n2 = inputs[1]->getDims()[inputsShapeDim - 1];

  // 如果进行转置，需要交换宽高
  if (transA) {
    std::swap(m1, n1);
  }
  if (transB) {
    std::swap(m2, n2);
  }

  // 第一个矩阵的高要等于第二个的宽
  IT_ASSERT(n1 == m2);

  // 最后两个之前的维度，取两个矩阵的最大值
  Shape ans = inputs[0]->getDims();
  for (int i = 0; i < inputsShapeDim - 2; i++) {
    ans[i] = std::max(ans[i], inputs[1]->getDims()[i]);
  }
  ans[ans.size() - 2] = m1;
  ans[ans.size() - 1] = n2;

  return {{ans}};
}

}  // namespace infini