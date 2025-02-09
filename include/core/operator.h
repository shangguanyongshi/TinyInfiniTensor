#pragma once

#include "core/op_type.h"
#include "core/tensor.h"

namespace infini {
using KernelAttrs = std::tuple<Device, OpType::underlying_t>;

class GraphObj;
class OperatorObj : public Object {
  friend class GraphObj;

 protected:
  OpType type; // 算子的类型
  TensorVec inputs;                        // 算子可能有多个输入，都保存在 inputs 中
  TensorVec outputs;                       // 算子可能有多个输出，都保存在 outputs 中
  vector<WRef<OperatorObj>> predecessors;  // 以 weak_ptr 的形式保存当前算子的所有前驱算子
  vector<WRef<OperatorObj>> successors;    // 以 weak_ptr 的形式保存当前算子的所有后继算子

 public:
  OperatorObj(OpType opType, TensorVec inputs, TensorVec outputs);
  
  /**
   * @brief 根据给定的输入张量推算当前算子所有输出张量的形状
   * @param inputs 当前算子所有输入张量
   * @return optional<vector<Shape>> 对于给定的 inputs 输入，当前算子运算后输出张量的形状
   */
  virtual optional<vector<Shape>> inferShape(const TensorVec &inputs) = 0;

  /**
   * @brief 获取给定输入的张脸数组中每个张量的类型
   * @param inputs 给定的输入张量数组
   * @return vector<DataType> 每个张量的类型
   */
  virtual vector<DataType> inferDataType(const TensorVec &inputs) const;

  /**
   * @brief Constructs outputs (if requried) and check whether the operator is valid.
   *
   * @param graph If graph is not nullptr, outputs should be created in this function.
   * @return true if the operator is valid.
   * @return false if the operator is invalid.
   */
  bool checkValid(GraphObj *graph);

 public:  // getter and setter
  /**
   * @brief 返回当前算子所有的输入张量
   * @return const TensorVec& 以常量引用的方式返回当前算子所有的输入张量
   */
  const TensorVec &getInputs() const { return inputs; }

  /**
   * @brief 返回当前算子所有的输出张量
   * @return const TensorVec& 以常量引用的方式返回当前算子所有的输出张量
   */
  const TensorVec &getOutputs() const { return outputs; }

  /**
   * @brief 返回第 i 个输入张量
   * @param i 输入张量的索引
   * @return Tensor 返回的第 i 个输入张量
   */
  Tensor getInputs(size_t i) const { return inputs.at(i); }

  /**
   * @brief 返回当前算子的第一个输出张量
   * @return Tensor 返回的第一个输出张量
   */
  Tensor getOutput() const {
    IT_ASSERT(outputs.size() == 1, "Unimplemented");
    return outputs[0];
  }

  /**
   * @brief 返回当前算子的第 i 个输出张量
   * @param i 输出张量的索引
   * @return Tensor 返回的第 i 个输出张量
   */
  Tensor getOutput(size_t i) const {
    IT_ASSERT(i < outputs.size(), "Index exceeded");
    return outputs.at(i);
  }

  /**
   * @brief 以 shared_ptr 的形式返回当前算子的所有前驱算子
   * @return OpVec 当前算子的所有前驱算子
   */
  OpVec getPredecessors() const { return wrefs_to_refs(predecessors); }

  /**
   * @brief 以 shared_ptr 的形式返回当前算子的所有后继算子
   * @return OpVec 当前算子的所有后继算子
   */
  OpVec getSuccessors() const { return wrefs_to_refs(successors); }

  OpType getOpType() const { return type; }
  // HACK: set correct data type
  DataType getDType() const { return getInputs(0)->getDType(); }
  DataType getOutDType() const { return getOutput()->getDType(); }
  virtual int numInputs() const = 0;
  virtual int numOutputs() const = 0;

  /**
   * @brief Clone this operator and replace its inputs and outputs.
   *
   * @param newInputs
   * @param newOutputs
   * @return Operator
   */
  virtual Operator clone(const TensorVec &newInputs, const TensorVec &newOutputs) const = 0;

 protected:
  /**
   * @brief 推算当前算子所有输出张量的形状
   * @return optional<vector<Shape>> 当前算子所有输出张量的形状
   */
  optional<vector<Shape>> inferShape();
  vector<DataType> inferDataType() const;

 private:
  void addPredecessors(const Operator &op) { predecessors.emplace_back(op); }
  void addSuccessors(const Operator &op) { successors.emplace_back(op); }
  void removePredecessors(const Operator &op);
  void removeSuccessors(const Operator &op);
  /**
   * @brief 用 t2 替换 t1
   * @param t1 源张量
   * @param t2 目标张量
   */
  void replaceInput(Tensor t1, Tensor t2);
};

#define OP_CLONE(OpObj)                                                                            \
  virtual Operator clone(const TensorVec &newInputs, const TensorVec &newOutputs) const override { \
    auto op = infini::make_ref<OpObj>(*this);                                                      \
    op->inputs = newInputs;                                                                        \
    op->outputs = newOutputs;                                                                      \
    op->predecessors.clear();                                                                      \
    op->successors.clear();                                                                        \
    IT_ASSERT(op->checkValid(nullptr));                                                            \
    return op;                                                                                     \
  }

}  // namespace infini
