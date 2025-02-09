#pragma once
#include "core/common.h"
#include "core/ref.h"

namespace infini {

class RuntimeObj;
using Runtime = Ref<RuntimeObj>;

/**
 * @brief Blob 类中有一个运行时和空指针成员，其模板成员函数 getPtr 用于将指针转换为指定类型并返回
 */
class BlobObj
{
  Runtime runtime;
  void *ptr;

public:
  BlobObj(Runtime runtime, void *ptr) : runtime(runtime), ptr(ptr) {}
  BlobObj(BlobObj &other) = delete;
  BlobObj &operator=(BlobObj const &) = delete;
  ~BlobObj() {};

  template <typename T>
  T getPtr() const { return reinterpret_cast<T>(ptr); }
};

} // namespace infini
