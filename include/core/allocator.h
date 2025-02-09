#pragma once
#include "core/runtime.h"
#include "core/tensor.h"
#ifdef BUILD_TEST
#include "gtest/gtest.h"
#endif
#include <cstddef>
#include <map>
#include <unordered_set>

namespace infini {
/**
 * @brief 包含模拟内存分配的 alloc 和 free 函数，以及实际内存分配的 getPtr 函数
 */
class Allocator {
 private:
  Runtime runtime;

  size_t used;  // 已经使用

  size_t peak;  // 保存最近一次 alloc 后所使用内存的最大值

  size_t alignment;  // 执行内存对齐时的参考大小

  // pointer to the memory actually allocated
  void *ptr;

  // =================================== 作业 ===================================
  // TODO：可能需要设计一个数据结构来存储free block，以便于管理和合并
  // HINT: 可以使用一个 map 来存储 free block，key 为 block 的起始/结尾地址，value 为 block 的大小
  // =================================== 作业 ===================================
  // 按照(内存块大小，内存块起始地址)保存已经使用过后并释放的空闲内存块，
  // 内部按以内存块大小排序减少分配内存时的查询时间
  std::map<size_t, size_t> freeBlocks;
  // 按照(内存块起始地址，内存块大小)保存已经使用过后并释放的空闲内存块，
  // 内部按内存块起始地址排序减少释放内存时的查询时间
  std::map<size_t, size_t> freeBlocksPos;

 public:
  Allocator(Runtime runtime);

  virtual ~Allocator();

  // function: simulate memory allocation
  // arguments：
  //     size: size of memory block to be allocated
  // return: head address offset of the allocated memory block
  /**
   * @brief alloc 和 free 配合模拟实际执行推理时内存的使用和释放过程，可用于预先优化内存的分配策略
   *        alloc 分配 size 大小的内存，在调用 getPtr 时， 再根据所有的模拟，执行实际的内存分配
   * @param size 此次需要分配的内存大小
   * @return size_t 返回当前分配的内存相对于起始地址的偏移量
   */
  size_t alloc(size_t size);

  // function: simulate memory free
  // arguments:
  //     addr: head address offset of memory block to be free
  //     size: size of memory block to be freed
  /**
   * @brief alloc 和 free 配合模拟实际执行推理时内存的使用和释放过程，可用于预先优化内存的分配策略
   *        free 释放之前分配的相对于内存起始偏移 addr 的 size 大小的内存
   * @param addr 之前分配的相对于内存起始地址的偏移量
   * @param size 此次需要释放的内存大小
   */
  void free(size_t addr, size_t size);

  // function: perform actual memory allocation
  // return: pointer to the head address of the allocated memory
  /**
   * @brief getPtr 用于根据之前调用 alloc 和 free 模拟时所需内存的峰值大小，
   *        执行实际的内存分配，并返回所分配内存的起始地址
   * @return void* 返回分配的内存的起始地址
   */
  void *getPtr();

  void info();

 private:
  // function: memory alignment, rouned up
  // return: size of the aligned memory block
  /**
   * @brief 小于等于 size 且是 alignment 整数倍的内存大小
   * @param size 需要对齐的内存大小
   * @return size_t 返回对齐后的内存大小
   */
  size_t getAlignedSize(size_t size);
};
}  // namespace infini
