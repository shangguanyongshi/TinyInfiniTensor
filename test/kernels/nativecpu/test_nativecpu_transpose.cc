#include "core/graph.h"
#include "core/kernel.h"
#include "core/runtime.h"
#include "operators/transpose.h"
#include "test.h"

namespace infini {

TEST(Transpose, NativeCpu) {
  Runtime runtime = NativeCpuRuntimeObj::getInstance();
  Graph g = make_ref<GraphObj>(runtime);
  
  // 指定转置维度
  Shape permute = {0, 2, 1, 3};
  
  // 为计算图添加一个输入张量
  auto input = g->addTensor({1, 2, 3, 4}, DataType::Float32);
  // 为计算图添加一个转置算子
  auto op = g->addOp<TransposeObj>(input, nullptr, permute);

  // 为计算图的所有张量分配内存
  g->dataMalloc();

  // 为输入张量设置数据
  input->setData(IncrementalGenerator());

  runtime->run(g);

  EXPECT_TRUE(op->getOutput(0)->equalData(
      vector<float>{0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23}));
}

}  // namespace infini
