#pragma once
#include <functional>
#include <memory>
#include <type_traits>

#include "core/common.h"

namespace infini {

template <typename T>
using Ref = std::shared_ptr<T>;
template <typename T>
using WRef = std::weak_ptr<T>;

// 以下创建了 is_ref 模板类型，并通过特化来判断是否是 Ref 或 WRef 类型
// 当传入的类型是 Ref 或 WRef 时，is_ref 的 value 为 true，否则为 false
template <typename T>
struct is_ref : std::false_type {};
template <typename T>
struct is_ref<Ref<T>> : std::true_type {};
template <typename T>
struct is_ref<WRef<T>> : std::true_type {};

/**
 * @brief 使用给定的参数创建 T 类型的 shared_ptr 指针
 * @tparam T 指针所管理的对象类型
 * @tparam Params 利用参数创建所管理的对象
 * @param params Params 的模板参数
 * @return Ref<T> 返回的 shared_ptr<T> 指针对象
 */
template <typename T, typename... Params>
Ref<T> make_ref(Params &&...params) {
  // 此处用于保证 T 不应该是 Ref 或 WRef 类型
  static_assert(is_ref<T>::value == false, "Ref should not be nested");
  return std::make_shared<T>(std::forward<Params>(params)...);
}

template <class T, class U, typename std::enable_if_t<std::is_base_of_v<U, T>> * = nullptr>
Ref<T> as(const Ref<U> &ref) {
  return std::dynamic_pointer_cast<T>(ref);
}

/**
 * @brief 利用 shared_ptr 元素构成的数组中的每个元素初始化一个 weak_ptr 元素构成的数组
 * @tparam T 智能指针所管理对象的类型
 * @param refs shared_ptr 元素组成的数组
 * @return std::vector<WRef<T>> refs 中对应元素转换为 weak_ptr 元素后的数组
 */
template <typename T>
std::vector<WRef<T>> refs_to_wrefs(const std::vector<Ref<T>> &refs) {
  std::vector<WRef<T>> wrefs;
  for (const auto &ref : refs) wrefs.emplace_back(ref);
  return wrefs;
}
/**
 * @brief 使用 weak_ptr 元素构成的数组中的每个元素初始化一个 shared_ptr 元素构成的数组（会使 weak_ptr 所关联的
 * shared_ptr 的引用计数 + 1）
 * @tparam T 智能指针所管理对象的类型
 * @param wrefs weak_ptr 元素组成的数组
 * @return std::vector<Ref<T>> wrefs 中对应元素转换为 shared_ptr 元素后的数组
 */
template <typename T>
std::vector<Ref<T>> wrefs_to_refs(const std::vector<WRef<T>> &wrefs) {
  std::vector<Ref<T>> refs;
  for (const auto &wref : wrefs) refs.emplace_back(wref);
  return refs;
}

}  // namespace infini
