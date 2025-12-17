# RadixCache v2 Improvements

## Overview

`radix_cache_v2.py` addresses the key limitations identified in the code review of `radix_cache.py` (v1). This document outlines the improvements and how they address the production-readiness concerns.

## Key Issues Addressed

### 1. ✅ Page Slot Semantics (Not Just Cache Indices)

**v1 Problem:**
- `device_indices` were arbitrary cache indices
- No connection to actual memory locations
- "索引共享" (index sharing) not "内存共享" (memory sharing)

**v2 Solution:**
- Introduced `PageSlot` class representing actual page slots
- `page_ids` point to real page slots managed by `PageAllocator`
- Indices are now first-class citizens with proper semantics
- Each page slot has metadata (ref_count, access_time, etc.)

### 2. ✅ Page Allocator with Free List / Compaction

**v1 Problem:**
- No page allocator
- No free list for reuse
- No compaction mechanism
- Memory grows unbounded

**v2 Solution:**
- `PageAllocator` class manages physical page slots
- Free list for reusing evicted pages
- `compact()` method for memory reclamation
- Bounded memory with `max_pages` limit
- Proper allocation/deallocation lifecycle

**Features:**
```python
allocator = PageAllocator(max_pages=10000, layer_idx=0)
page_id = allocator.allocate(ref_count=1)  # Allocate new page
allocator.deallocate(page_id)  # Return to free list
allocator.compact()  # Reclaim memory
```

### 3. ✅ Reference Counting for Shared Prefixes

**v1 Problem:**
- No ref-count semantics
- Shared prefixes not properly tracked
- No way to know when a page can be safely evicted

**v2 Solution:**
- `PageSlot.ref_count` tracks number of requests sharing the page
- `increment_ref()` / `decrement_ref()` methods
- Automatic deallocation when ref_count reaches 0
- `can_evict()` checks if page is safe to evict

**Usage:**
```python
# When matching prefix, increment ref counts
for page_id in matched_page_ids:
    allocator.increment_ref(page_id)

# When request completes, decrement
for page_id in request_page_ids:
    allocator.decrement_ref(page_id)
```

### 4. ✅ Eviction Policy (LRU / LFU / Hybrid)

**v1 Problem:**
- Only had `hit_count` and `last_access_time` statistics
- No actual eviction behavior
- No memory pressure handling

**v2 Solution:**
- `EvictionPolicy` enum (LRU, LFU, HYBRID)
- `evict_pages()` method with policy-based selection
- `get_eviction_score()` calculates eviction priority
- Automatic eviction when memory pressure occurs

**Eviction Policies:**
- **LRU (Least Recently Used)**: Evict oldest accessed pages
- **LFU (Least Frequently Used)**: Evict least frequently accessed pages
- **HYBRID**: Combination of both (70% age, 30% frequency)

**Usage:**
```python
cache = RadixCacheV2(eviction_policy=EvictionPolicy.LRU)
evicted_pages = cache.evict(num_pages=100, layer_idx=0)
```

## Architecture Improvements

### Page Slot Lifecycle

```
Allocate → Use → Increment Ref → Decrement Ref → Can Evict → Deallocate → Free List
```

### Reference Counting Flow

```
Request 1 matches prefix → increment_ref(page_id) → ref_count = 1
Request 2 matches same prefix → increment_ref(page_id) → ref_count = 2
Request 1 completes → decrement_ref(page_id) → ref_count = 1
Request 2 completes → decrement_ref(page_id) → ref_count = 0 → can_evict()
```

### Eviction Flow

```
Memory pressure → evict_pages(num_pages) → 
  Get evictable pages (ref_count == 0) → 
  Sort by eviction_score → 
  Evict top N → 
  Deallocate → 
  Add to free_list
```

## API Changes

### MatchResult
- `page_ids` (new): Tensor of actual page slot IDs
- `device_indices` (alias): For backward compatibility

### RadixCacheV2.insert()
- Now requires `page_ids` tensor (from page allocator)
- Automatically handles reference counting
- Returns number of tokens inserted

### New Methods
- `evict(num_pages, layer_idx)`: Evict pages based on policy
- `compact(layer_idx)`: Compact free list
- `get_stats(layer_idx)`: Get allocator statistics

## Comparison: v1 vs v2

| Feature | v1 | v2 |
|---------|----|----|
| Page semantics | Arbitrary cache indices | Actual page slots |
| Page allocator | ❌ | ✅ |
| Free list | ❌ | ✅ |
| Compaction | ❌ | ✅ |
| Reference counting | ❌ | ✅ |
| Eviction policy | ❌ | ✅ (LRU/LFU/Hybrid) |
| Memory bounds | Unbounded | Bounded (max_pages) |
| Production ready | ❌ | ✅ |

## Usage Example

```python
from radix_cache_v2 import RadixCacheV2, RadixKey, EvictionPolicy

# Create cache with eviction policy
cache = RadixCacheV2(
    device="cuda",
    page_size=1,
    max_pages=10000,
    eviction_policy=EvictionPolicy.LRU
)

# Match prefix
key = RadixKey(token_ids=[1, 2, 3, 4, 5])
match_result = cache.match_prefix(key, layer_idx=0)
print(f"Matched {len(match_result.page_ids)} pages")

# Insert new prefix (requires page_ids from allocator)
allocator = cache._get_allocator(layer_idx=0)
page_ids = torch.tensor([allocator.allocate() for _ in range(5)])
cache.insert(key, page_ids, layer_idx=0)

# Evict pages when needed
evicted = cache.evict(num_pages=100, layer_idx=0)

# Get statistics
stats = cache.get_stats(layer_idx=0)
print(f"Active pages: {stats['active_pages']}")
```

## Migration Notes

To migrate from v1 to v2:

1. Replace `RadixCache` with `RadixCacheV2`
2. Update `match_prefix()` calls to include `layer_idx`
3. Update `insert()` calls to provide `page_ids` from allocator
4. Add reference counting when matching/inserting
5. Add eviction logic when memory pressure occurs

## Conclusion

RadixCache v2 is a **production-ready** implementation that addresses all the key limitations of v1:

- ✅ Proper page slot semantics
- ✅ Page allocator with free list and compaction
- ✅ Reference counting for shared prefixes
- ✅ LRU/LFU eviction policies
- ✅ Bounded memory usage

This makes it suitable for production use cases where memory management and eviction are critical.
