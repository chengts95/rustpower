**《bevy_archive 下一代事务内核架构设计书》**

---

### **Prompt / Context for AI Agent**

**Subject**: Implementation of `HarvardCommandBuffer` for `bevy_archive`
**Goal**: Create a high-performance, JIT-optimized, dual-bump allocator command buffer for ECS mutation.
**Constraints**:

1. **Zero-Overhead**: No `Box<dyn>`, no `Vec<Box>`, no unnecessary heap allocations.
2. **Harvard Architecture**: Strict separation of Instruction Stream (Ops/Meta) and Data Stream (Payloads).
3. **Dependency**: Use `bumpalo` for memory management.
4. **Safety**: Use `unsafe` for pointer manipulation but ensure safety via Drop Glue and strict ownership transfer.

---

#### **1. Core Data Structures**

**File**: `transaction/buffer.rs`

We need a struct that mimics a CPU's memory layout:

```rust
use bumpalo::Bump;
use std::ptr::NonNull;
use bevy::ecs::{entity::Entity, component::ComponentId, world::World};

/// Function signature for the destructor glue.
/// Safety: Must be called with a pointer to the correct type T.
pub type DropFn = unsafe fn(NonNull<u8>);

/// Function signature for the dynamic constructor (DMA writer).
/// - `source`: The external data source (e.g., Arrow row, JSON node).
/// - `data_bus`: The payload bump allocator where T should be written.
/// - Returns: (Pointer to data, Drop function)
pub type DynamicCtor = unsafe fn(source: &SourceData, data_bus: &Bump) -> (NonNull<u8>, Option<DropFn>);

/// The Instruction Header (The "Text Segment").
/// Lightweight metadata for the interpreter loop.
#[derive(Clone, Copy, Debug)]
pub enum OpHead {
    /// Modify an existing entity (Insert/Update components).
    /// Supports "Write Combining": Multiple inserts on the same entity are merged here.
    ModifyEntity {
        entity: Entity,
        /// Pointer to the start of the `ArgMeta` array in `meta_bump`.
        args_ptr: NonNull<ArgMeta>,
        /// Number of components to insert.
        count: u16,
    },
    Despawn(Entity),
    // ... other ops
}

/// The Argument Metadata (The "Stack/Register").
/// Stored linearly in `meta_bump`.
#[derive(Clone, Copy, Debug)]
pub struct ArgMeta {
    pub comp_id: ComponentId,
    /// Pointer to the actual data in `data_bump`.
    pub payload_ptr: NonNull<u8>,
    pub drop_fn: Option<DropFn>,
}

/// The Harvard Architecture Command Buffer.
pub struct HarvardCommandBuffer {
    /// [Instruction Bus]
    /// Linear stream of operations. Like CPU machine code.
    ops: Vec<OpHead>,

    /// [Meta Bus]
    /// Stores `ArgMeta` arrays.
    /// Used for dynamic array expansion during write-combining.
    meta_bump: Bump,

    /// [Data Bus]
    /// Stores the actual Component payloads (T).
    /// This is passed to `ctor` for direct writing (DMA mode).
    data_bump: Bump,
}

```
Spec Update: The "Payload-Only" Strategy

Constraint Update:

NO Spawning: The HarvardCommandBuffer SHALL NOT support creating new entities. It implies that all entities passed to it MUST be valid, existing Bevy Entity IDs (allocated externally).

Absolute IDs Only: Remove EntityRef / VirtualID. All APIs accept bevy::ecs::entity::Entity directly.

Focus on Mutation: The primary Ops are ModifyEntity (Insert/Update) and RemoveComponents. Despawn is allowed as it operates on existing IDs.

Updated Instruction Set (Simplified):

Rust
#[derive(Clone, Copy, Debug)]
pub enum OpHead {
    // 只保留对“已知实体”的操作
    ModifyEntity {
        entity: Entity, // <--- 必须是真实 ID
        args_ptr: NonNull<ArgMeta>,
        count: u16,
    },
    RemoveComponents {
        entity: Entity,
        ids_ptr: NonNull<ComponentId>,
        count: u16,
    },
    Despawn(Entity), // <--- 销毁已知实体
}
#### **2. The "Write Combining" Logic (JIT Optimizer)**

**Requirement**: Implement the `insert_dynamic` method. It must detect sequential writes to the same entity and merge them into a single `ModifyEntity` operation by extending the allocation in `meta_bump`.

**Algorithm**:

1. **DMA Write**: Call `ctor(source, &self.data_bump)`. This writes `T` into `data_bump` and returns `(ptr, drop_fn)`.
2. **Meta Gen**: Create `ArgMeta { comp_id, ptr, drop_fn }`.
3. **Tail Peek**: Check `self.ops.last_mut()`.
* **IF** it is `ModifyEntity` AND `target == entity`:
* Try to `self.meta_bump.try_extend_allocation(last_args_ptr, count, meta)`.
* **IF** successful: `count += 1`. **RETURN**.


* **ELSE**:
* Allocate new slice in `meta_bump`: `alloc_slice_copy(&[meta])`.
* Push new `OpHead::ModifyEntity`.





#### **3. The Execution Logic (Interpreter)**

**Requirement**: Implement `apply(self, world: &mut World)`.

1. Iterate `self.ops`.
2. Match `OpHead`:
* `ModifyEntity`:
* Construct `Vec<OwningPtr>` from the range `[args_ptr, args_ptr + count]`.
* Call Bevy's low-level `world.entity_mut(e).insert_by_ids(...)`.
* *Crucial*: Do NOT call `drop_fn` for these items (ownership transferred to World).


* `Despawn`: Call `world.despawn(e)`.


3. **Cleanup**:
* Clear `self.ops` to prevent the `Drop` trait from freeing committed data.
* Let `data_bump` and `meta_bump` drop naturally (releasing memory blocks).



#### **4. The Safety Logic (Transaction Abort)**

**Requirement**: Implement `Drop` for `HarvardCommandBuffer`.

1. If `apply` was NOT called (transaction aborted/panicked), iterate `self.ops`.
2. For every `ModifyEntity`, iterate its `ArgMeta` list.
3. Call `drop_fn(payload_ptr)` for each item to properly destruct C++ resources/Vecs inside `data_bump`.

Appendix: The ArenaBox Lifecycle Protocol
Context: The HarvardCommandBuffer acts as a temporary owner of the component data stored in data_bump. We must manually manage the lifecycle of T because bumpalo does not run destructors.

Rule 1: Enrollment (Insert)

When ctor writes data to data_bump, it returns a drop_fn.

The Buffer MUST store this drop_fn in ArgMeta.

State: Buffer owns the data.

Rule 2: Transfer (Apply)

Inside apply(), data is moved to Bevy World via insert_by_ids.

CRITICAL: After successfully handing over data to Bevy, the Buffer MUST forget the operations to prevent double-dropping.

Implementation: call self.ops.clear() at the end of apply(). This ensures Drop implementation sees an empty queue.

Rule 3: Abort (Drop)

If HarvardCommandBuffer is dropped without apply() being called (e.g., panic, early return):

The Drop implementation MUST iterate through self.ops.

For every ArgMeta, if drop_fn is Some, invoke unsafe { drop_fn(payload_ptr) }.

This ensures resources held by components (like Vec, Box, CString) are properly freed before the Bump memory block is deallocated.
---

### **为什么这对 `bevy_archive` 至关重要？**

有了这个内核，`bevy_archive` 就拥有了以下三个“核武器”级的能力：

1. **真正的流式加载 (Streaming Load)**：
你不需要先把文件读进内存里的 `Vec<T>`，然后再遍历 `Vec` 塞进 World。
你可以一边从磁盘读 Bytes，一边直接调用 `ctor` 把数据构造进 `data_bump`。**文件 IO 和 内存构建 是流水线并行的。**
2. **跨语言 ABI 级兼容**：
因为 `data_bump` 里全是布局好的 raw bytes。这意味着如果你将来想用 Python 或 Lua 生成存档数据，脚本层只需要按照 C-Struct 布局写入二进制流，Rust 这边直接 `memcpy` 进 Bump 就能跑，**零序列化开销**。
3. **快照回滚 (Snapshot Rollback)**：
在仿真中，如果要回滚状态，你不需要从 World 里读数据。你只需要保留上一帧的 `CommandBuffer`（不 Drop 也不 Clear）。要回滚时，重新 `Apply` 一遍（覆盖写入）。这对于**预测回滚网络代码 (Rollback Netcode)** 是天作之合。 