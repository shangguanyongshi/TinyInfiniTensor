#include "core/runtime.h"

#include <chrono>
#include <cstring>
#include <memory>

#include "core/blob.h"
#include "core/graph.h"
#include "core/kernel.h"
namespace infini {
void NativeCpuRuntimeObj::run(const Graph &graph) const {
  // 获取搜索 kernel 的单例
  const auto &kernelRegistry = KernelRegistry::getInstance();

  // 依次遍历每个算子，搜索对应的 kernel 执行算子对应的运算
  for (auto &op : graph->getOperators()) {
    // 根据算子的类型和设备类型，设置所需要的 kernel 属性
    auto kernelAttrs = KernelAttrs{device, op->getOpType().underlying()};
    // 获取对应的 kernel
    Kernel *kernel = kernelRegistry.getKernel(kernelAttrs);
    // 利用 kernel 进行计算
    kernel->compute(op, this);
  }
}

string NativeCpuRuntimeObj::toString() const { return "CPU Runtime"; }

void NativeCpuRuntimeObj::dealloc(void *ptr) { return free(ptr); }

void *NativeCpuRuntimeObj::alloc(size_t size) {
  // size + sizeof(uint64_t) - 1 使得分配的空间大于等于 size 且是 uint64_t 的整数倍
  return calloc((size + sizeof(uint64_t) - 1) / sizeof(uint64_t), sizeof(uint64_t));
}

}  // namespace infini
