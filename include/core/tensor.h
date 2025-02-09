#pragma once
#include <cmath>
#include <cstring>
#include <fstream>

#include "core/blob.h"
#include "core/data_type.h"
#include "core/object.h"
#include "core/runtime.h"

namespace infini {
class GraphObj;
using ShapeElem = int;
using Shape = vector<ShapeElem>;
class TensorObj : public Object {
  friend class GraphObj;

 protected:
  int dim;  // 保存 shape 的维度

  DataType dtype;  // 张量中元素的数据类型
  vector<WRef<OperatorObj>> targets;  // 保存所有引用该 Tensor 的 weak_ptr 类型的 Operator
  WRef<OperatorObj> source;  // 保存生成该 Tensor 的 weak_ptr 类型的 Operator
  Blob data; // 保存 Tensor 中实际的数据（只有一个 runtime 和 ptr 成员，ptr 成员指向保存实际数据的内存）
  Runtime runtime; // 保存 Tensor 的运行时

 private:
  Shape shape;   // 保存 Tensor 的 shape
  size_t _size;  // 保存 Tensor 所有维度的乘积
  
  /**
   * @brief clone 出来的张量应该具有相同的 Fuid。
   * Cloned tensors share the same id. Tensors constructed from scratch have a new id.
   */
  Fuid fuid;

 public:
  /**
   * @brief 根据给定的形状和数据类型构造张量
   * @param shape 张量的形状
   * @param dtype 张量的数据类型
   * @param runtime 张量的运行时
   */
  TensorObj(Shape shape, DataType dtype, Runtime runtime);
  virtual ~TensorObj() {}

  /**
   * @brief 将 Tensor 的信息以字符串形式返回
   * @return string Tensor 的信息字符串
   */
  string toString() const override;

  /**
   * @brief 返回张量中所有值的个数
   * @return size_t 张量中的值个数
   */
  size_t size() const { return _size; }

  // 返回 Tensor 中数据所占的字节数
  size_t getBytes() const { return _size * dtype.getSize(); }

  /**
   * @brief 获取张量的形状
   * @return Shape 当前张量的形状
   */
  Shape getDims() const { return shape; }

  /**
   * @brief 设置当前 Tensor 的 shape（会同步更新与 shape 相关的信息）
   * @param shape_ 新的 shape
   */
  void setShape(Shape shape_);

  /**
   * @brief 获取 Tensor 的维度
   * @return size_t Tensor 的维度
   */
  size_t getRank() const { return shape.size(); }
  /**
   * @brief 获取 Tensor 的 Family unique ID
   * @return UidBaseType Fuid 类型重载了转换为 UidBaseType
   * 类型的操作符，因此直接返回 UidBaseType 类型
   */
  UidBaseType getFuid() const { return fuid; }

  void setData(std::function<void(void *, size_t, DataType)> const &generator) const;

  void setDataBlob(const Blob &blob);

  void printData() const;

  /**
   * @brief 比较给定的张量是否完全相同
   * @param rhs 待比较的张量
   * @param relativeError 误差
   * @return true 
   * @return false 
   */
  bool equalData(const Tensor &rhs, double relativeError = 1e-6) const;

  template <typename T>
  bool equalData(const vector<T> &dataVector) {
    IT_ASSERT(size() == dataVector.size());
    IT_ASSERT(DataType::get<T>() == dtype.cpuTypeInt());
    return equalDataImpl(getRawDataPtr<T *>(), dataVector.data(), size());
  }

  /**
   * @brief 返回 Tensor 中 Blob data 的内部指针
   * @tparam T Blob data 的内部指针需要转换到的指针类型
   * @return T 转换为 T 类型后的指针
   */
  template <typename T>
  T getRawDataPtr() const {
    static_assert(std::is_pointer_v<T>,
                  "Raw data pointer has a type of pointer");
    IT_ASSERT(data != nullptr);
    return data->getPtr<T>();
  }

  DataType getDType() const { return dtype; }
  Runtime getRuntime() const { return runtime; }

  /**
   * @brief 以 weak_ptr 列表的形式返回所有引用该 Tensor 的 Operator
   * @return OpVec 所有引用该 Tensor 的 Operator，以 weak_ptr 的形式返回
   */
  OpVec getTargets() const { return wrefs_to_refs(targets); }

  /**
   * @brief 以 shared_ptr 形式返回生成该 Tensor 的 Operator
   * @return Operator shared_ptr 类型的生成该张量的 Operator
   */
  Operator getSource() const { return source.lock(); }

 private:
  /**
   * @brief 将 Tensor 的数据转换为字符串表示
   *
   * 该函数模板将 Tensor 的数据转换为字符串表示。
   * 1. 包含 Tensor 的唯一标识符。然后，它获取 Tensor 的维度数量，并初始化
   * 一个大小为维度数量的向量，其初始值为 1。接着，通过调用 getPtr<T *>() 获取指向
   * 数据的指针，并设置向量的最后一个元素为 Tensor 形状的最后一个维度大小。
   * 
   * 函数通过一个循环从后向前计算每个维度的大小，并存储在向量中。在遍历 Tensor 数据的
   * 主循环中，函数根据向量的值来确定何时插入左方括号 [ 和右方括号 ]，以表示多维数组的结构。
   * 每个数据元素通过指针访问，并追加到字符串流中。如果当前元素不是最后一个元素，则在其后
   * 添加逗号和空格。最后，如果当前元素位于列的末尾，则插入换行符。
   * 
   * @tparam T 数据类型
   * @return string Tensor 的数据字符串表示
   */
  template <class T>
  string dataToString() const {
    std::stringstream builder;
    builder << "Tensor: " << guid << std::endl;

    auto numDims = shape.size();
    auto dimSzVec = vector<int>(numDims, 1);
    auto ptr = data->getPtr<T *>();
    dimSzVec[numDims - 1] = shape[numDims - 1];

    for (int i = numDims - 1; i != 0; --i)
      dimSzVec[i - 1] = dimSzVec[i] * shape[i - 1];

    for (size_t i = 0, iEnd = size(); i < iEnd; ++i) {
      for (size_t j = 0; j < numDims; ++j)
        if (i % dimSzVec[j] == 0) builder << "[";

      builder << ptr[i];
      for (size_t j = 0; j < numDims; ++j)
        if ((int)i % dimSzVec[j] == dimSzVec[j] - 1) builder << "]";

      if (i != size() - 1) builder << ", ";

      auto column = (size_t)dimSzVec[numDims - 1];
      if (i % column == column - 1) builder << std::endl;
    }
    return builder.str();
  }
  
  /**
   * @brief 比较两个数组的数据是否相等
   *
   * 该函数模板用于比较两个数组 a 和 b 中的元素是否相等。它根据模板参数 T 的类型
   * 进行不同的处理。如果 T 是整数类型，则直接比较两个数组对应位置的元素是否相等。
   * 如果 T 是浮点类型，则使用相对误差进行比较。
   *
   * @tparam T 数组元素的类型
   * @param a 指向第一个数组的指针
   * @param b 指向第二个数组的指针
   * @param size 数组的大小
   * @param relativeError 相对误差，默认为 1e-6
   * @return bool 如果两个数组在给定的误差范围内相等，则返回 true；否则返回 false
   */
  template <typename T>
  bool equalDataImpl(const T *a, const T *b, size_t size,
                     double relativeError = 1e-6) const {
    for (size_t i = 0; i < size; ++i) {
      if constexpr (std::is_integral_v<T>) {
        if (a[i] != b[i]) return false;
      } else if constexpr (std::is_floating_point_v<T>) {
        if (std::min(fabs(a[i]), fabs(b[i])) == 0. &&
            fabs(a[i] - b[i]) > relativeError) {
          printf("Error on %lu: %f %f\n", i, a[i], b[i]);
          return false;
        } else if (std::min(fabs(a[i]), fabs(b[i])) != 0. &&
                   fabs(a[i] - b[i]) / std::max(fabs(a[i]), fabs(b[i])) >
                       relativeError) {
          printf("Error on %lu: %f %f\n", i, a[i], b[i]);
          return false;
        }
      } else {
        static_assert(!sizeof(T), "Unsupported data type");
      }
    }
    return true;
  }

  /**
   * @brief 以 weak_ptr 的格式添加使用该张量的算子
   * @param op 使用该张量的算子
   */
  void addTarget(const Operator &op) { targets.emplace_back(op); }

  /**
   * @brief 以 weak_ptr 的格式添加生成该张量的算子
   * @param op 
   */
  void setSource(const Operator &op) { source = op; }

  /**
   * @brief 标识给定的算子不再使用当前张量了，从记录列表中删除
   * @param op 要删除的算子
   */
  void removeTarget(const Operator &op) {
    for (auto itr = targets.begin(); itr != targets.end();) {
      if (itr->lock() == op)
        itr = targets.erase(itr);
      else
        ++itr;
    }
  }
};

}  // namespace infini
