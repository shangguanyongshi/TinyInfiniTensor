#pragma once
#include "core/common.h"
#include "ref.h"

namespace infini {

using UidBaseType = int;

class Uid {
 private:
  UidBaseType uid;

 public:
  Uid(UidBaseType uid) : uid(uid) {}
  Uid &operator=(const Uid &rhs) = delete;

  operator UidBaseType() const { return uid; }
};

/**
 * @brief 用于唯一标记一个张量
 */
class Guid : public Uid {
 private:
  UidBaseType generateGuid() {
    static UidBaseType guidCnt = 0;
    return ++guidCnt;
  }

 public:
  Guid() : Uid(generateGuid()) {}
  Guid(const Guid &rhs) : Uid(generateGuid()) {}
};

/**
 * @brief 用来标记所有 clone 得到的张量，它们应该具有相同的 Fuid。
 * Family unique ID. Cloned tensors shared the same FUID.
 */
class Fuid : public Uid {
 private:
  UidBaseType generateFuid() {
    static UidBaseType fuidCnt = 0;
    return ++fuidCnt;
  }

 public:
  Fuid() : Uid(generateFuid()) {}
  Fuid(const Fuid &fuid) : Uid(fuid) {}
};

/**
 * @brief 有一个 Guid 成员变量，用于唯一标识对象
 */
class Object {
 protected:
  Guid guid;

 public:
  virtual ~Object() {};
  virtual string toString() const = 0;
  void print() { std::cout << toString() << std::endl; }
  UidBaseType getGuid() const { return guid; }
};

inline std::ostream &operator<<(std::ostream &os, const Object &obj) {
  os << obj.toString();
  return os;
}

// Overload for Ref-wrapped Object
template <typename T, typename std::enable_if_t<std::is_base_of_v<Object, T>> * = nullptr>
inline std::ostream &operator<<(std::ostream &os, const Ref<T> &obj) {
  os << obj->toString();
  return os;
}

}  // namespace infini
