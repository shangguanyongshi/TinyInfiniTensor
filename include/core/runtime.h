#pragma once
#include "core/common.h"
#include "core/op_type.h"
#include "core/ref.h"

namespace infini {
class TensorObj;
class OperatorObj;
class GraphObj;
class RuntimeObj;
class BlobObj;

using Tensor = Ref<TensorObj>;
using Operator = Ref<OperatorObj>;
using Graph = Ref<GraphObj>;
using Runtime = Ref<RuntimeObj>;
using Blob = Ref<BlobObj>;

using TensorVec = vector<Tensor>;
using OpVec = vector<Operator>;

enum class Device { CPU = 1 };

/**
 * @brief 运行时对象的抽象类，定义了运行时对象应该包含的成员函数
 */
class RuntimeObj : public std::enable_shared_from_this<RuntimeObj> {
 protected:
  Device device;

 public:
  explicit RuntimeObj(Device device) : device(device) {}
  RuntimeObj(RuntimeObj &other) = delete;
  RuntimeObj &operator=(RuntimeObj const &) = delete;
  virtual ~RuntimeObj() {}

  /**
   * @brief 遍历所传入 graph 中的所有算子，利用 KernelRegistry 查找对应的 kernel，并执行计算
   * @param graph 要推理运行的计算图
   */
  virtual void run(const Graph &graph) const = 0;
  virtual void *alloc(size_t size) = 0;
  virtual void dealloc(void *ptr) = 0;

  bool isCpu() const { return true; }

  /**
   * @brief 返回当前运行时类的名称（不同的子类可以自定义返回的名称）
   * @return string 表示名称的字符串
   */
  virtual string toString() const = 0;
};

/**
 * @brief 基于运行时抽象类，实现适用于 CPU 的运行时类，其 run 函数会遍历 graph 中的所有算子，
 * 利用 KernelRegistry 查找对应的 kernel，并执行计算
 */
class NativeCpuRuntimeObj : public RuntimeObj {
 public:
  NativeCpuRuntimeObj() : RuntimeObj(Device::CPU) {}

  /**
   * @brief 定义为单例类型
   * @return Ref<NativeCpuRuntimeObj>& 返回 shared_ptr<NativeCpuRuntimeObj> 类型的对象
   */
  static Ref<NativeCpuRuntimeObj> &getInstance() {
    static Ref<NativeCpuRuntimeObj> instance = make_ref<NativeCpuRuntimeObj>();
    return instance;
  }

  /**
   * @brief 释放 ptr 所指向的内存
   * @param ptr 指向要释放内存的指针
   */
  void dealloc(void *ptr) override;

  /**
   * @brief 遍历所传入 graph 中的所有算子，利用 KernelRegistry 查找对应的 kernel，并执行计算
   * @param graph 要推理运行的计算图
   */
  void run(const Graph &graph) const override;

  /**
   * @brief 分配 size 大小的内存空间（会通过运算，保证空间大于等于 size 且是 uint64_t 的整数倍）
   * @param size 要分配内存的最小值
   * @return void* 返回分配的内存空间的指针
   */
  void *alloc(size_t size) override;

  /**
   * @brief 返回当前运行时类的名称
   * @return string 表示名称的字符串
   */
  string toString() const override;
};

}  // namespace infini
