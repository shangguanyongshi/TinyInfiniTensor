#pragma once
#include "core/common.h"
#include <functional>
#include <memory>
#include <type_traits>

namespace infini {

template <typename T> using Ref = std::shared_ptr<T>;
template <typename T> using WRef = std::weak_ptr<T>;

// 以下创建了 is_ref 模板类型，并通过特化来判断是否是 Ref 或 WRef 类型
// 当传入的类型是 Ref 或 WRef 时，is_ref 的 value 为 true，否则为 false
template <typename T> struct is_ref : std::false_type {};
template <typename T> struct is_ref<Ref<T>> : std::true_type {};
template <typename T> struct is_ref<WRef<T>> : std::true_type {};

/**
 * @brief 使用给定的参数创建 T 类型的 shared_ptr 指针
 * @tparam T 指针所管理的对象类型
 * @tparam Params 利用参数创建所管理的对象
 * @param params Params 的模板参数
 * @return Ref<T> 返回的 shared_ptr<T> 指针对象
 */
template <typename T, typename... Params> Ref<T> make_ref(Params &&...params) {
    // 此处用于保证 T 不应该是 Ref 或 WRef 类型
    static_assert(is_ref<T>::value == false, "Ref should not be nested");
    return std::make_shared<T>(std::forward<Params>(params)...);
}

template <class T, class U,
          typename std::enable_if_t<std::is_base_of_v<U, T>> * = nullptr>
Ref<T> as(const Ref<U> &ref) {
    return std::dynamic_pointer_cast<T>(ref);
}

template <typename T>
std::vector<WRef<T>> refs_to_wrefs(const std::vector<Ref<T>> &refs) {
    std::vector<WRef<T>> wrefs;
    for (const auto &ref : refs)
        wrefs.emplace_back(ref);
    return wrefs;
}

template <typename T>
std::vector<Ref<T>> wrefs_to_refs(const std::vector<WRef<T>> &wrefs) {
    std::vector<Ref<T>> refs;
    for (const auto &wref : wrefs)
        refs.emplace_back(wref);
    return refs;
}

} // namespace infini
