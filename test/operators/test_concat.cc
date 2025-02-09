#include "core/graph.h"
#include "core/runtime.h"
#include "operators/concat.h"
#include "test.h"

namespace infini {
TEST(Concat, ShapeInfer) {
  Runtime runtime = NativeCpuRuntimeObj::getInstance();
  Graph g = make_ref<GraphObj>(runtime);
  auto t1 = g->addTensor({1, 3, 2, 4}, DataType::Float32);
  auto t2 = g->addTensor({1, 3, 2, 5}, DataType::Float32);

  auto op = g->addOp<ConcatObj>(TensorVec{t1, t2}, nullptr, 3);
  EXPECT_EQ(op->getOutput()->getDims(), (Shape{1, 3, 2, 9}));
}

TEST(Concat, ShapeInfer2) {
  Runtime runtime = NativeCpuRuntimeObj::getInstance();
  Graph g = make_ref<GraphObj>(runtime);

  auto t1 = g->addTensor({2, 2, 3, 1}, DataType::Float32);
  auto t2 = g->addTensor({2, 2, 1, 1}, DataType::Float32);
  auto t3 = g->addTensor({2, 2, 2, 1}, DataType::Float32);

  auto op = g->addOp<ConcatObj>(TensorVec{t1, t2, t3}, nullptr, 2);
  
  EXPECT_EQ(op->getOutput()->getDims(), (Shape{2, 2, 6, 1}));
}
}  // namespace infini
