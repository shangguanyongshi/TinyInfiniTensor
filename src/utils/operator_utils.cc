#include "utils/operator_utils.h"

#include "core/runtime.h"

namespace infini {

Shape infer_broadcast(const Shape &A, const Shape &B) {
  // =================================== 作业 ===================================
  // TODO：对 A 和 B 进行双向广播，返回广播后的形状。
  // REF: https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md
  // =================================== 作业 ===================================
  size_t rank1 = A.size();
  size_t rank2 = B.size();
  // 保证 A 的 rank 大于等于 B 的 rank
  if (rank1 < rank2) {
    return infer_broadcast(B, A);
  }
  if (rank2 == 0) {
    // 如果 B 张量的 rank 为 0，直接返回 A 的形状
    return A;
  }

  // 以较大的 rank 作为结果的 rank
  Shape ans(rank1, 1);

  if (rank1 != rank2) {
    // B 的尺寸比 A 小，将 B 后面的尺寸复制到 ans 中
    for (size_t i = 0; i < rank2; ++i) {
      ans[rank1 - rank2 + i] = B[i];
    }
  }

  // 从前向后遍历 A 和 ans 的尺寸，选择最大的尺寸作为结果的尺寸
  for (size_t i = 0; i < rank1; ++i) {
    if (ans[i] != A[i]) {
      // 如果 A 和 ans 的尺寸不相等，判断是否有一个为 1 的情况
      if (ans[i] == 1 || A[i] == 1) {
        // 如果有一个为 1，选择较大的尺寸作为结果的尺寸
        ans[i] = std::max(ans[i], A[i]);
      } else {
        // 如果两个尺寸都没有一个为 1，抛出异常
        IT_ASSERT(0);
      }
    }
  }

  return ans;
}

int get_real_axis(const int &axis, const int &rank) {
  IT_ASSERT(rank >= 1);
  IT_ASSERT(axis >= -rank && axis <= (rank - 1));
  int newAxis;
  if (axis < 0) {
    newAxis = rank + axis;
  } else {
    newAxis = axis;
  }
  return newAxis;
}

Shape locate_index(size_t inputN, const Shape &shape) {
  Shape ans(shape.size());
  auto i = ans.rbegin();
  auto j = shape.rbegin(), ej = shape.rend();
  while (j != ej) {
    auto div = std::div(inputN, *j++);
    *i++ = div.rem;
    inputN = div.quot;
  }
  return ans;
}

size_t delocate_index(const Shape &shapeIndex, const Shape &shape, const Shape &stride) {
  size_t ans = 0;
  Shape index(shapeIndex.size());
  IT_ASSERT(shapeIndex.size() == shape.size());
  IT_ASSERT(shape.size() == stride.size());
  for (size_t i = 0; i < shape.size(); ++i) {
    index[i] = shapeIndex[i] % shape[i];
    ans += index[i] * stride[i];
  }
  return ans;
}

std::string device_to_str(Device device) {
  std::string deviceStr;
  switch (device) {
    case Device::CPU:
      return "CPU";
    default:
      IT_TODO_HALT();
  }
}

std::string get_kernel_attrs_str(const KernelAttrs &kernelAttrs) {
  std::string deviceStr = device_to_str(std::get<0>(kernelAttrs));
  std::string opStr = OpType(std::get<1>(kernelAttrs)).toString();
  return deviceStr + ", " + opStr;
}

}  // namespace infini
