#include "core/allocator.h"

#include <utility>

namespace infini {
Allocator::Allocator(Runtime runtime) : runtime(runtime) {
  used = 0;
  peak = 0;
  ptr = nullptr;

  // 'alignment' defaults to sizeof(uint64_t), because it is the length of
  // the longest data type currently supported by the DataType field of
  // the tensor
  alignment = sizeof(uint64_t);
}

Allocator::~Allocator() {
  if (this->ptr != nullptr) {
    runtime->dealloc(this->ptr);
  }
}

size_t Allocator::alloc(size_t size) {
  IT_ASSERT(this->ptr == nullptr);
  // pad the size to the multiple of alignment
  size = this->getAlignedSize(size);

  // =================================== 作业 ===================================
  // TODO: 设计一个算法来分配内存，返回起始地址偏移量
  // =================================== 作业 ===================================

  // 查找是否有空闲的块大于等于 size 的大小
  auto biggerBlock = freeBlocks.lower_bound(size);
  if (biggerBlock != freeBlocks.end()) {
    // 如果有空闲块大于等于 size 的大小，直接分配
    size_t allocAddr = biggerBlock->second;
    size_t blockSize = biggerBlock->first;
    freeBlocks.erase(biggerBlock);
    freeBlocksPos.erase(allocAddr);
    // 如果空闲块大于 size，将剩余部分重新插入 freeBlocks
    if (blockSize > size) {
      freeBlocks.insert({blockSize - size, allocAddr + size});
      freeBlocksPos.insert({allocAddr + size, blockSize - size});
    }
    this->used += size;
    return allocAddr;
  }
  // 如果没有空闲块可以容纳 size 大小的内存块，是否有空闲内存紧邻峰值内存块部分
  if (!freeBlocksPos.empty()) {
    size_t lastBlockAddr = freeBlocksPos.rbegin()->first;
    size_t lastBlockSize = freeBlocksPos.rbegin()->second;
    if (lastBlockAddr + lastBlockSize == this->peak) {
      // 如果最后一个空闲块紧邻峰值内存块，在最后一个内存块的基础上扩展
      size_t allocAddr = lastBlockAddr;
      freeBlocks.erase(lastBlockSize);
      freeBlocksPos.erase(lastBlockAddr);
      this->used += size;
      this->peak += size - lastBlockSize;
      return allocAddr;
    }
  }
  // 没有空闲块包含峰值内存块直接在当前峰值内存块后分配新的内存块
  size_t allocAddr = this->peak;
  this->peak += size;
  this->used += size;
  return allocAddr;
}

void Allocator::free(size_t addr, size_t size) {
  IT_ASSERT(this->ptr == nullptr);
  size = getAlignedSize(size);

  // =================================== 作业 ===================================
  // TODO: 设计一个算法来回收内存
  // =================================== 作业 ===================================
  if (freeBlocks.empty()) {
    // 如果没有空闲块，直接插入
    freeBlocks.insert({size, addr});
    freeBlocksPos.insert({addr, size});
    this->used -= size;
    return;
  }
  // 分别记录是否要和前一个或后一个空闲块合并
  bool isMergeWithPre = false;
  bool isMergeWithNext = false;
  size_t preBlockAddr = 0;
  size_t preBlockSize = 0;
  size_t nextBlockAddr = 0;
  size_t nextBlockSize = 0;
  // 查找 addr 是否与某个空闲块相邻，如果相邻则合并
  // 先找到第一个内存地址大于 addr 的空闲内存块
  auto geqBlock = freeBlocksPos.upper_bound(addr);
  if (geqBlock == freeBlocksPos.begin()) {
    // 如果空闲地址的起始地址都大于 addr，判断能否将要释放的地址与第一个空闲地址合并
    if (geqBlock->first == addr + size) {
      // addr + size 与第一个空闲地址相同，合并
      nextBlockAddr = geqBlock->first;
      nextBlockSize = geqBlock->second;
      isMergeWithNext = true;
    }
  }
  if (geqBlock == freeBlocksPos.end()) {
    // 如果空闲地址的起始地址都小于 addr，判断能否将要释放的地址与最后一个空闲地址合并
    auto lastBlock = freeBlocksPos.end();
    lastBlock--;
    if (lastBlock->first + lastBlock->second == addr) {
      // addr 与最后一个空闲地址相同，合并
      preBlockAddr = lastBlock->first;
      preBlockSize = lastBlock->second;
      isMergeWithPre = true;
    }
  }
  // 如果 addr 位于两个空闲块之间，判断能否与两个空闲块合并
  if (geqBlock != freeBlocksPos.end() && geqBlock != freeBlocksPos.begin()) {
    auto preBlock = geqBlock;
    preBlock--;  // 前一个空闲块
    auto nextBlock = geqBlock;
    nextBlock++;  // 后一个空闲块
    if (preBlock->first + preBlock->second == addr && nextBlock->first == addr + size) {
      // addr 与前后两个空闲地址相邻，将三个空闲块合并
      preBlockAddr = preBlock->first;
      preBlockSize = preBlock->second;
      nextBlockAddr = nextBlock->first;
      nextBlockSize = nextBlock->second;
      isMergeWithPre = true;
      isMergeWithNext = true;
    } else if (preBlock->first + preBlock->second == addr) {
      // addr 与前一个空闲地址相同，合并
      preBlockAddr = preBlock->first;
      preBlockSize = preBlock->second;
      isMergeWithPre = true;
    } else if (nextBlock->first == addr + size) {
      // addr 与后一个空闲地址相同，合并
      nextBlockAddr = nextBlock->first;
      nextBlockSize = nextBlock->second;
      isMergeWithNext = true;
    }
  }

  // 执行实际的内存释放
  if (isMergeWithPre) {
    // 释放的内存与前一个空闲块合并
    freeBlocks.erase(preBlockSize);
    freeBlocksPos.erase(preBlockAddr);
    freeBlocks.insert({preBlockSize + size, preBlockAddr});
    freeBlocksPos.insert({preBlockAddr, preBlockSize + size});
  }
  if (isMergeWithNext) {
    // 释放的内存与后一个空闲块合并
    freeBlocks.erase(nextBlockSize);
    freeBlocksPos.erase(nextBlockAddr);
    freeBlocks.insert({size + nextBlockSize, addr});
    freeBlocksPos.insert({addr, size + nextBlockSize});
  }

  if (!isMergeWithPre && !isMergeWithNext) {
    // 释放的内存不与任何空闲块合并
    freeBlocks.insert({size, addr});
    freeBlocksPos.insert({addr, size});
  }
  this->used -= size;
}

void *Allocator::getPtr() {
  if (this->ptr == nullptr) {
    // 根据模拟时所需内存的峰值大小，执行实际的内存分配
    this->ptr = runtime->alloc(this->peak);
    printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
  }
  return this->ptr;
}

size_t Allocator::getAlignedSize(size_t size) {
  // 计算小于等于 size 且是 alignment 整数倍的内存大小
  return ((size - 1) / this->alignment + 1) * this->alignment;
}

void Allocator::info() { std::cout << "Used memory: " << this->used << ", peak memory: " << this->peak << std::endl; }
}  // namespace infini
