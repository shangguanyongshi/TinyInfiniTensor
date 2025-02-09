#include "core/graph.h"
#include "core/runtime.h"
#include "operators/element_wise.h"
#include "test.h"

namespace infini {

using ExpectOutput = vector<float>;

/**
 * @brief 用指定的数据和答案，测试模板参数指定的二元算子的正确性
 * @tparam T 要测试的算子类型
 * @param generator1 生成第一个张量数据的函数
 * @param generator2 生成第二个张量数据的函数
 * @param shape1 第一个张量数据的形状
 * @param shape2 第二个张量数据的形状
 * @param ansVec 运行后期望得到的结果
 */
template <class T>
void testElementWiseNativeCpu(const std::function<void(void *, size_t, DataType)> &generator1,
                              const std::function<void(void *, size_t, DataType)> &generator2,
                              const Shape &shape1,
                              const Shape &shape2,
                              const ExpectOutput &ansVec) {
  Runtime runtime = NativeCpuRuntimeObj::getInstance();
  Graph g = make_ref<GraphObj>(runtime);
  // 添加两个张量
  auto t1 = g->addTensor(shape1, DataType::Float32);
  auto t2 = g->addTensor(shape2, DataType::Float32);
  
  // 添加模板参数指定的算子类型
  auto op = g->addOp<T>(t1, t2, nullptr);

  // 对所有张量分配内存
  g->dataMalloc();

  // 为两个张量设置数据
  t1->setData(generator1);
  t2->setData(generator2);

  // 运行计算图
  runtime->run(g);
  EXPECT_TRUE(op->getOutput()->equalData(ansVec));
}

TEST(ElementWise, NativeCpu) {
  testElementWiseNativeCpu<AddObj>(IncrementalGenerator(), 
                                   IncrementalGenerator(),
                                   Shape{1, 2, 2, 3, 1},
                                   Shape{2, 1, 1},
                                   ExpectOutput{0, 1, 2, 4, 5, 6, 6, 7, 8, 10, 11, 12});
  testElementWiseNativeCpu<MulObj>(IncrementalGenerator(),
                                   IncrementalGenerator(),
                                   Shape{1, 2, 2, 3, 1},
                                   Shape{2, 1, 1},
                                   ExpectOutput{0, 0, 0, 3, 4, 5, 0, 0, 0, 9, 10, 11});
  testElementWiseNativeCpu<SubObj>(IncrementalGenerator(),
                                   IncrementalGenerator(),
                                   Shape{1, 2, 2, 3, 1},
                                   Shape{2, 1, 1},
                                   ExpectOutput{0, 1, 2, 2, 3, 4, 6, 7, 8, 8, 9, 10});
  testElementWiseNativeCpu<DivObj>(IncrementalGenerator(),
                                   OneGenerator(),
                                   Shape{1, 2, 2, 3, 1},
                                   Shape{2, 1, 1},
                                   ExpectOutput{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
}

}  // namespace infini
