#include "core/graph.h"

#include <algorithm>
#include <numeric>
#include <queue>

#include "operators/matmul.h"
#include "operators/transpose.h"

namespace infini {

void GraphObj::addOperatorAndConnect(const Operator &op) {
  sorted = false;
  // 先加入保存所有算子的 vector 中
  ops.push_back(op);
  // 处理当前算子的所有输入张量
  for (auto &input : op->getInputs()) {
    if (input) {
      // 将当前算子的输入张量的目标张量设置为当前算子
      input->addTarget(op);
      // 如果当前张量有源算子，同时更新当前算子和源算子的前驱和后继算子
      if (auto pred = input->getSource()) {
        pred->addSuccessors(op);
        op->addPredecessors(pred);
      }
    }
  }
  // 处理当前算子的所有输出张量
  for (auto &output : op->getOutputs()) {
    if (output) {
      // 将当前算子设置为当前张量的源算子
      output->setSource(op);
      // 如果当前张量有目标算子，同时更新当前算子和目标算子的前驱和后继算子
      for (auto &succ : output->getTargets()) {
        succ->addPredecessors(op);
        op->addSuccessors(succ);
      }
    }
  }
}

string GraphObj::toString() const {
  std::ostringstream oss;
  oss << "Graph Tensors:\n";
  for (const auto &tensor : tensors) oss << tensor << "\n";

  oss << "Graph operators:\n";
  for (const auto &op : ops) {
    vector<UidBaseType> preds, succs;
    for (auto &o : op->getPredecessors()) preds.emplace_back(o->getGuid());
    for (auto &o : op->getSuccessors()) succs.emplace_back(o->getGuid());
    oss << "OP " << op->getGuid();
    oss << ", pred " << vecToString(preds);
    oss << ", succ " << vecToString(succs);
    oss << ", " << op << "\n";
  }
  return oss.str();
}

bool GraphObj::topo_sort() {
  if (this->sorted) {
    return true;
  }
  std::vector<Operator> sorted;             // 保存排好序的节点
  std::unordered_set<OperatorObj *> flags;  // 记录已经被排好序的算子
  sorted.reserve(ops.size());
  flags.reserve(ops.size());
  while (sorted.size() < ops.size()) {
    // 标记是否有算子被加入排序后的 sorted，计算图中有环或 ops 为空时，为
    // false，Any node is move to sorted in this loop.
    auto modified = false;
    for (auto const &op : ops) {
      // 如果当前算子的所有前继算子都已经被排序，表示可以将当前张量加入到排序列表中
      if (auto const &inputs = op->getInputs();   // 获取当前算子的所有输入张量
          flags.find(op.get()) == flags.end() &&  // 如果当前算子还没有被排序
          std::all_of(inputs.begin(),  // 并且当前算子的所有前继算子都已经被排序
                      inputs.end(),
                      [&flags](auto const &input) {
                        // 获取生成当前张量的算子
                        auto ptr = input->getSource().get();
                        // 如果当前张量没有生成算子，或者生成算子已经被排好序，返回 true
                        return !ptr || flags.find(ptr) != flags.end();
                      })) {
        modified = true;
        sorted.emplace_back(op);
        flags.insert(op.get());
      }
    }
    if (!modified) {
      return false;
    }
  }
  // 将排好序的算子列表重新移动给 ops
  this->ops = std::move(sorted);
  // 标记当前计算图已经排好序
  return this->sorted = true;
}

void GraphObj::optimize() {
  // =================================== 作业 ===================================
  // TODO: 设计一个算法来实现指定的图优化规则
  // 图优化规则如下：
  // 1. 去除冗余的算子（例如，两个相邻的算子都是 transpose 算子，且做的是相反的操作，可以将其全部删除）
  // 2.合并算子（例如，矩阵乘算子中含有属性transA、transB，如果其输入存在transpose，且对最后两个维度做交换，
  //   就可以将transpose融入到矩阵乘算子的属性中去）
  // =================================== 作业 ===================================

  // 记录是否要继续优化
  bool continueOptimize = true;

  while (continueOptimize) {
    // 先标记不再优化了，如果此次做了优化，再其置为 true
    continueOptimize = false;

    // 先执行拓扑排序
    IT_ASSERT(topo_sort() == true);
    
    // 遍历所有算子
    for (auto &op : ops) {
      // 如果是 transpose 算子，获取其后继算子
      if (op->getOpType() == OpType::Transpose) {
        // 获取当前算子的所有后继算子
        auto successors = op->getSuccessors();

        // 如果当前 transpose 只有一个 transpose 后继算子
        if (successors.size() == 1 &&
            successors[0]->getOpType() == OpType::Transpose) {
          // 比较这两个算子的 permute 属性是否相同
          TransposeObj *tran1 = dynamic_cast<TransposeObj *>(op.get());
          TransposeObj *tran2 = dynamic_cast<TransposeObj *>(successors[0].get());
          if (tran1->getPermute() == tran2->getPermute()) {
            // 如果相同，将 op 的输入张量作为 tran2 后继算子的输入

            // 获取 op 的输入张量
            Tensor opInput = op->getInputs()[0];

            // 如果 opInput 有源算子，从其后继节点列表中删除 op 算子
            if (opInput->getSource()) {
              opInput->getSource()->removeSuccessors(op);
            }
            // 删除 opInput 的唯一 target 算子
            opInput->removeTarget(op);
            
            // 修改 trans2 所有后继算子的前驱和原有的输入(trans2 的输出)：
            // 1. 将这些后继节点中 trans2 的输出张量替换为 op 的输入张量
            // 2. 将 trans2 从这些后继算子的前驱算子列表中删除
            // 3. 修改 opInput 的目标算子
            // 4. 将这些后继节点的前驱节点设置为 op 的前驱节点
            // 5. 将这些后继节点添加到 op 前驱节点的后继节点列表中
            auto trans2Successors = successors[0]->getSuccessors();
            for (auto &trans2Successor : trans2Successors) {
              trans2Successor->replaceInput(successors[0]->getOutputs()[0], opInput);
              trans2Successor->removePredecessors(successors[0]);
              opInput->addTarget(trans2Successor);
              if (opInput->getSource()) {
                trans2Successor->addPredecessors(opInput->getSource());
                opInput->getSource()->addSuccessors(trans2Successor);
              }
            }

            // 删除 op 的输出张量
            removeTensor(op->getOutputs()[0]);
            // 删除 trans2 的输出张量
            removeTensor(successors[0]->getOutputs()[0]);
            // 删除两个算子
            removeOperator(op);
            removeOperator(successors[0]);

            // 标记需要继续优化
            continueOptimize = true;
            break;
          }
        }

        // 如果 transpose 只有一个 matmul 后继算子，将 transpose 操作合并到 matmul 
        if (successors.size() == 1 &&
            successors[0]->getOpType() == OpType::MatMul) {
          
          // 置反 matmul 对应 op 输出的 tran
          // 1. 查找 op 的输出张量是 matmul 的第几个输入
          int transIndex = -1;
          for (int i = 0; i < successors[0]->numInputs(); ++i) {
            if (successors[0]->getInputs(i) == op->getOutputs()[0]) {
              transIndex = i;
              break;
            }
          }
          IT_ASSERT(transIndex != -1);
          // 2. 置反该输入的 transpose 操作
          MatmulObj *matmul = dynamic_cast<MatmulObj *>(successors[0].get());
          if (transIndex == 0) {
            matmul->setTransA(!matmul->getTransA());
          } else {
            matmul->setTransB(!matmul->getTransB());
          }

          // 获取 op 的输入张量
          Tensor opInput = op->getInputs()[0];
          
          // 从 opInput 的 targets 中删除 op
          opInput->removeTarget(op);
          // 将 matmul 添加到 opInput 的 target
          opInput->addTarget(successors[0]);

          // 从 op 后继的前继列表中删除 op
          successors[0]->removePredecessors(op);
          // 利用 opInput 替换 op 后继中 op 的输出
          successors[0]->replaceInput(op->getOutputs()[0], opInput);

          if (opInput->getSource()) {
            // 从 opInput 的前继列表中删除 op
            opInput->getSource()->removeSuccessors(op);
            // 从 opInput 的前继列表中添加 op 的后继
            opInput->getSource()->addSuccessors(successors[0]);
            // 将 opInput 的源设置为 mul 的前继
            successors[0]->addPredecessors(opInput->getSource());
          }
          
          // 删除 op 的输出张量
          removeTensor(op->getOutputs()[0]);
          // 删除 op
          removeOperator(op);

          // 标记需要继续优化
          continueOptimize = true;
          break;
        }
      }
    }
  }
}

Tensor GraphObj::getTensor(int fuid) const {
  for (auto tensor : tensors) {
    if (tensor->getFuid() == fuid) {
      return tensor;
    }
  }
  return nullptr;
}

void GraphObj::shape_infer() {
  for (auto &op : ops) {
    // 获取推断出的所有输出张量的形状
    auto ans = op->inferShape();
    IT_ASSERT(ans.has_value());
    // 保存所有的输出张量
    auto oldOutputs = op->getOutputs();
    // 推断出的输出张量维度的个数应该和输出张量个数相同
    IT_ASSERT(ans.value().size() == oldOutputs.size());
    // 如果推断出的输出张量形状和当前算子中的对应输出张量的形状不同，根据新形状修改当前计算图中的对应输出张量的形状
    // （算子中对应输出张量的形状没有修改）
    // replace the old outputshape and size with new one
    for (int i = 0; i < (int)ans.value().size(); ++i) {
      // 获取推断出的第 i 个输出张量的形状
      auto newShape = ans.value()[i];
      // 获取第 i 个输出张量的形状
      auto oldShape = oldOutputs[i]->getDims();
      auto fuid = oldOutputs[i]->getFuid();
      if (newShape != oldShape) {
        // 获取当前计算图中与第 i
        // 个输出张量对应的张量（将算子中的输出张量拷贝到当前计算图中时，他们的
        // fuid 相同）
        auto tensor = this->getTensor(fuid);
        // 重置当前对应输出张量的形状
        tensor->setShape(newShape);
      }
    }
  }
}

void GraphObj::dataMalloc() {
  // topological sorting first
  IT_ASSERT(topo_sort() == true);

  // =================================== 作业 ===================================
  // TODO：利用 allocator 给计算图分配内存
  // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
  // =================================== 作业 ===================================

  // 记录每个张量所分配内存地址的偏移量
  std::unordered_map<TensorObj *, size_t> tensorAddrOffsets;

  // 记录每个输入张量被多少个算子使用（以在模拟分配时确定是否能释放该张量的内存）
  std::unordered_map<TensorObj *, size_t> inputUsedCount;

  // 1. 遍历拓扑排序后的计算图中的所有算子，根据输入和输出张量的形状，模拟计算过程中内存的使用情况
  //   - InfiniTensor 中输入和输出张量预先执行了内存的模拟 alloc，此处是否需要？
  //     只需要给输入张量预分配内存即可，这样下面模拟时可以直接从第一个算子的输出开始，只执行 [为输出分配]-> [释放输入]

  // 1.1 先遍历所有张量，为输入张量预分配内存，同时计算每个输入张量被多少算子使用
  for (auto &tensor : tensors) {
    // 如果当前张量没有生成算子，即为输入张量，为其分配内存
    if (tensor->getSource() == nullptr) {
      tensorAddrOffsets[tensor.get()] = allocator.alloc(tensor->getBytes());
    }
    // 计算当前张量有多少个算子使用
    if (tensor->getTargets().size() != 0) {
      inputUsedCount[tensor.get()] = tensor->getTargets().size();
    }
  }
  
  // 1.2 遍历算子执行模拟
  for (auto &op : ops) {
    // 获取算子的所有输出张量，执行内存分配
    auto outputs = op->getOutputs();
    for (auto &output : outputs) {
      tensorAddrOffsets[output.get()] = allocator.alloc(output->getBytes());
    }
    // 获取算子的所有输入张量，释放内存
    auto inputs = op->getInputs();
    for (auto &input : inputs) {
      // 先递减使用当前张量的算子个数
      inputUsedCount[input.get()]--;
      // 如果当前张量没有被任何算子使用，释放内存
      if (inputUsedCount[input.get()] == 0) {
        allocator.free(tensorAddrOffsets[input.get()], input->getBytes());
        // 从记录张量的使用 map 中删除当前张量，提高后续的搜索效率
        inputUsedCount.erase(input.get());
      }
    }
  }

  // 2. 根据模拟的结果，执行实际的内存分配（alloctor.getPtr），并将分配好的内存位置记录到对应张量的 data.ptr 中
  for (auto &tensor : tensors) {
    tensor->setDataBlob(
        make_ref<BlobObj>(runtime, static_cast<uint8_t *>(allocator.getPtr()) + tensorAddrOffsets[tensor.get()]));
  }

  allocator.info();
}

Tensor GraphObj::addTensor(Shape dim, DataType dtype) {
  return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
}

Tensor GraphObj::addTensor(const Tensor &tensor) {
  IT_ASSERT(tensor->getRuntime() == runtime, std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                                                 tensor->getRuntime()->toString() + " to " + runtime->toString());
  tensors.emplace_back(tensor);
  return tensor;
}

TensorVec GraphObj::addTensor(const TensorVec &tensors) {
  for (auto &t : tensors) addTensor(t);
  return tensors;  // 此处 return 的是实参 tensors
}

// tensor's "source" and "target" must be in "ops".
// tensor has no "source" and no "target" must not exist.
// "inputs" or "outputs" of operators must be in "tensors"
// "predecessors" and "successors" of an operator of "ops" must be in "ops".
bool GraphObj::checkValid() const {
  for (auto tensor : tensors) {
    // 1. 张量的目标算子列表和源算子不能同时为空
    IT_ASSERT(!(tensor->getTargets().size() == 0 && nullptr == tensor->getSource()));

    for (auto op : tensor->getTargets()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
    }
    auto op = tensor->getSource();
    IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
  }
  for (auto op : ops) {
    for (auto tensor : op->getInputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) != tensors.end());
    }
    for (auto tensor : op->getOutputs()) {
      IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) != tensors.end());
    }
    for (auto pre : op->getPredecessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
    }
    for (auto suc : op->getSuccessors()) {
      IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
    }
  }
  std::set<UidBaseType> s;
  // check whether two tensors with the same FUID exist
  for (auto tensor : tensors) {
    int cnt = s.count(tensor->getFuid());
    IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
    s.insert(tensor->getFuid());
  }
  return true;
}

}  // namespace infini