# KV cache manager: a code walkthrough

[Guide index](README.md) · [Architecture](architecture.md) · [Source index](api-index.md) · [Foundations](foundations.md)

Draft walkthrough of [`KVCacheManager`](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L155) at `62bfced`. It covers all seven explicitly defined methods and properties. Diagram revisions and execution-order assumptions still require review; see [review notes](review-notes.md).

## Responsibilities

For this walkthrough, we will focus on two cooperating parts of the scheduling system:

| Component | Main responsibility |
| --- | --- |
| Scheduler | Decide which requests run and how many tokens each processes in a step. |
| KV cache manager | Track reusable KV blocks and allocate, share, or release blocks for requests. |

The `Request` class represents one request. In the code walkthrough, we will connect its fields to the request's tokens, processing progress, and block ownership, using the actual names and comments in the class.

In simple terms, the KV cache manager answers: **Can this request get the blocks it needs?** Some may already contain a matching prefix. Others need to be assigned for new KV data.

The scheduler uses that answer to decide whether the request can run now or whether it needs to wait or another request needs to be preempted.

## Free blocks can retain useful KV data

When a request finishes, we do not need to erase all of its cached data immediately.

A block can be free for allocation while still containing valid KV data from a previous request. Keeping that data gives a later request a chance to reuse it.

For the design described here, a block can be in one of three useful states:

| State | Can it be assigned to a different prefix? | Can a matching prefix reuse its existing data? |
| --- | --- | --- |
| Referenced by an active request | No, while it remains referenced. | Yes, if sharing is supported and the block is safe to share. |
| Unreferenced, with valid cached data | Yes; assignment will discard the old cached entry. | Yes, until the block is reassigned. |
| Unreferenced, with no valid cached data | Yes. | No. |

When multiple requests share a block, finishing one request does not make the block free. The block becomes available for reassignment only when no request references it. A reference count is one way to track this.

For an unreferenced cached block, we keep its hash-to-block mapping searchable and put the block on the LRU free list. If a later request matches that prefix, the manager takes the block off the free list and references it again.

If memory is needed for different tokens, the manager can select a block from the least recently used end of the free list. Before reassigning it, the manager removes the old hash mapping. The old KV data is no longer reusable and will be overwritten by the new computation.

The physical memory can remain inside the engine's allocated GPU block pool. Eviction here means losing the old cached contents, rather than necessarily returning GPU memory to the runtime.

### Diagram 3: a cached block's lifecycle

![A cached block moving between active use, the free cached queue, and reassignment to a different prefix.](assets/diagram-03-cached-block-lifecycle.png)

When the last reference is released, the block enters the back of the free list, shown as the most recently used end on the right. Its cached data and prefix lookup remain available. A matching request can reclaim that block from anywhere in the list and add an active reference without recomputing its KV data.

When storage is needed for different tokens, allocation selects the front of the list, shown as the least recently used end on the left. The manager removes the selected block's old prefix mapping before reassigning it. The new model computation will overwrite the old KV data.



## Memory pressure and preemption

Reusing a free block and preempting a running request are different operations.

If a block is on the free list, the manager can assign it without interrupting an active request. That block may still hold an older prefix, but no active request depends on it.

If there are no free blocks, the scheduler may need to preempt a running request to make progress.

Consider three running requests: A, B, and C. A needs another block to continue, but no blocks are free. Under a policy that selects the last request in the running queue, the scheduler chooses C as the victim.

The sequence is:

1. Pause C and retain its token sequence so it can resume later.
2. Release C's references to its KV blocks.
3. Make blocks with no remaining references available for allocation.
4. Assign a suitable available block to A so it can continue.
5. Reschedule C later and rebuild any KV data it no longer has.

Blocks still shared with another request cannot be reclaimed just by pausing C. If C does not release enough blocks, the scheduler must take another action according to its policy.

When C returns, prefix caching may reduce its recomputation. The manager looks for a still-available matching prefix, and C recomputes the missing suffix. But a hit is not guaranteed: A or another request may have overwritten C's old blocks in the meantime.

This is the trade-off: preemption lets other requests continue under memory pressure, but the paused request may wait longer and repeat model computation.

Choosing C because it is last in the running queue is a **scheduler policy**. Choosing an unused cached block from the LRU free list is a **cache replacement policy**. These should be explained separately in the code walkthrough.

### Diagram 4: preempt C so A can continue

![Request C is preempted to free KV capacity for A, then later resumes using any surviving cached prefix and recomputing missing KV data.](assets/diagram-04-request-preemption.png)

A needs another block, but the free list is empty. The scheduler pauses C and releases all of C's block references. Blocks still referenced by another request stay in use; blocks whose counts reach zero enter the free list. A can then claim available storage and continue. Reassigning a block removes its old prefix mapping, and the new computation overwrites its contents.

C's token sequence is retained, but `free_request` clears its page table. A surviving prefix therefore means KV data still available in shared blocks or in free cached blocks that have not been reassigned. When C is scheduled again, it must reclaim any available matching prefix and recompute the missing suffix. If no cached prefix survives, it must recompute from the start. Keeping token IDs alone does not preserve their KV data.



## Method walkthrough

The seven methods below cover initialization, capacity reporting, hash generation, prefix lookup, allocation, registration, and release. Each subsection follows the actual code and a worked example.

## Initialization

Method: `__init__` · [Source](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L230)

The constructor sets up the bookkeeping needed to manage KV cache blocks. It records which blocks are in use, which prefix each cached block represents, and which blocks are available for allocation.

Here is the method with the comments reworded for clarity. The logic is unchanged. `OrderedDict` is imported from Python's `collections` module.

```python
def __init__(self,
             num_blocks: int,
             block_size: int,
             enable_caching: bool):

    self.block_size = block_size
    self.enable_caching = enable_caching

    # Number of active references to each physical block.
    self.ref_cnt = [0] * num_blocks

    # Prefix hash represented by each physical block, if any.
    self.block_hash: list[int | None] = [None] * num_blocks

    # Available blocks, ordered from front to back for LRU reuse.
    # Block 0 is reserved for padding positions masked during attention.
    self.free: OrderedDict[int, None] = OrderedDict(
        (i, None) for i in range(1, num_blocks)
    )

    # Prefix hash -> physical block containing the corresponding KV data.
    self.hash_to_block: dict[int, int] = {}
```

### Block size and the caching flag

`block_size` is the number of token positions that fit in one KV block. For example, a block size of four means that each block can hold KV data for four tokens. It does not mean four bytes or four requests.

`enable_caching` records whether prefix caching is enabled. Disabling prefix caching does not mean disabling the KV cache used during generation. It means disabling prefix reuse through the caching mechanism.

In this constructor, the flag is only stored. All the bookkeeping structures are initialized regardless of its value. We will see how the flag affects behavior in the later methods.

### Reference counts: is a block still in use?

`ref_cnt` has one entry for every physical block. The list index is the block ID, and its value records how many active references that block has.

For example, if requests A and B share block 2, its reference count is two. When A releases it, the count becomes one. B still needs the block, so it cannot be reassigned. Only after B also releases it does the count become zero.

A zero count means there are no active references. For an ordinary usable block, that makes it eligible to return to the free list. It may still contain useful cached KV data.

The count tells us **how many references exist**, not which requests hold them. Request identities and their block assignments must be tracked elsewhere.

At initialization, every count is zero because the manager has not assigned any blocks yet. Block 0 also has a zero count, but it is reserved and excluded from normal allocation.

### Block hashes: what prefix does this block represent?

`block_hash` also has one entry per physical block. Given a block ID, it tells us which prefix hash is associated with the block's cached contents.

For example, `block_hash[2] = h` would mean that physical block 2 represents the cached prefix identified by `h`. As discussed earlier, the intended hash includes the preceding prefix as well as the current block's tokens. The constructor itself does not calculate those hashes.

Every entry starts as `None` because no cached prefix has been registered yet. `None` means there is no registered prefix hash for that block; it does not tell us whether the block has active references. That is the job of `ref_cnt`.

### The free list: which blocks can we allocate?

`free` contains the IDs of blocks available for allocation. It uses an `OrderedDict` so we can preserve their order and remove a block by its ID.

The keys are block IDs. The values are `None` because this structure only needs to track membership and order; it does not store KV tensors.

For the allocation and release behavior described earlier, we need three operations:

| Operation | How an `OrderedDict` supports it | Why we need it |
| --- | --- | --- |
| Take the front block | `popitem(last=False)` | Select the oldest available candidate for allocation. |
| Remove a particular block | `pop(block_id)` | Take a cached prefix block out of the free list when a request reuses it. |
| Append a released block at the back | `free[block_id] = None`, when the ID is absent | Return an unreferenced block as the newest available entry. |

These operations take constant time on average. `OrderedDict` combines lookup by key with maintained order; describing it only as a doubly linked list leaves out the lookup needed to remove an arbitrary block efficiently.

The constructor initializes the queue in increasing block-ID order. At this point, the blocks have no usage history. The later allocation, cache-hit, and release methods must maintain the order that makes it work as an LRU free list.

When a released block still contains a cached prefix, returning it to this list does not have to remove its prefix mapping. A later request may take it back through a prefix hit before it is reassigned.

### Block 0 and the number of usable blocks

The expression `range(1, num_blocks)` includes 1 and excludes `num_blocks`. Therefore, this code treats `num_blocks` as the **total number of physical blocks, including block 0**.

For example, with `num_blocks = 5`:

| Item | Initial value |
| --- | --- |
| Physical block IDs | 0, 1, 2, 3, 4 |
| Reserved block | 0 |
| `ref_cnt` | `[0, 0, 0, 0, 0]` |
| `block_hash` | `[None, None, None, None, None]` |
| `free`, from front to back | 1, 2, 3, 4 |
| `hash_to_block` | `{}` |
| Usable block count | 4 |

If we want **N usable blocks**, we must pass `num_blocks = N + 1`. Then the physical IDs run from 0 through N, inclusive, and the usable IDs run from 1 through N.

Block 0 is the null block used for **padding tokens**, as the code comment states. The padding positions are masked during attention, so their values do not contribute to the result. This reserved block is separate from the previous-block hash used to identify prefixes.

### Hash-to-block lookup: where is a cached prefix?

`hash_to_block` lets us look up a physical block using a prefix hash. For example, `hash_to_block[h] = 2` would tell us that the KV data associated with `h` is in block 2.

It complements `block_hash`:

| Structure | Lookup direction | Question it answers |
| --- | --- | --- |
| `block_hash` | Block ID → prefix hash | What cached prefix does this block represent? |
| `hash_to_block` | Prefix hash → block ID | Where can I find this cached prefix? |

The forward lookup helps find prefix hits. The per-block hash lets the manager identify the old cache entry when a block is reassigned. Later methods must keep these structures consistent.

`hash_to_block` starts empty because there are no registered cached prefixes. It stores block IDs, not the KV data itself.

### State after initialization

The manager now has zero references for every block, no registered prefixes, and an ordered collection of all usable blocks. The method initializes Python bookkeeping structures; it does not allocate the GPU tensors that hold the actual keys and values.

### Diagram 5: the manager's bookkeeping

![The KV cache manager immediately after initialization, with block 0 reserved and blocks 1 through 4 available for allocation.](assets/diagram-05-manager-bookkeeping.png)

`num_blocks = 5` includes the reserved padding block 0. All reference counts start at zero, all per-block hashes are `None`, and the prefix lookup is empty. The four usable blocks enter `free` in order: allocation takes from the front, while released blocks return at the back.

## Counting free blocks

Method: `num_free` · [Source](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L256)

This property returns the number of blocks currently available for allocation:

```python
@property
def num_free(self):
    return len(self.free)
```

The `@property` decorator lets us read it as `manager.num_free`, without parentheses. The value comes directly from the current free list; there is no separate counter to update.

With the earlier `num_blocks = 5` example, the initial free list contains blocks 1, 2, 3, and 4, so `num_free` returns four. Reserved block 0 is not included.

A free block may still contain valid cached KV data. This property counts every block available for allocation, including those cached blocks. It does not count prefix hits, identify requests, or change any state. Reading the length takes constant time.

## Updating block hashes

Method: `_update_hashes` · [Source](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L259)

This method extends `req.block_hashes` so the request has one hash for each complete block of token IDs. It leaves existing hashes in place and adds only the ones that are missing.

Here is the code with the comments condensed. The logic is unchanged:

```python
def _update_hashes(self, req: Request):
    """Add a prefix hash for each newly completed block of token IDs."""
    bs = self.block_size

    while len(req.block_hashes) < req.num_tokens // bs:
        i = len(req.block_hashes)
        parent = req.block_hashes[-1] if req.block_hashes else None
        req.block_hashes.append(
            hash((parent, tuple(req.token_ids[i * bs:(i + 1) * bs])))
        )
```

### Count the complete blocks

First, `bs` stores the block size. The expression `req.num_tokens // bs` uses integer division to count how many complete blocks the request's token sequence contains.

Use the example from the comments: nine tokens, represented here by letters, and a block size of four.

| Logical block index | Tokens | Complete? | Hash needed? |
| --- | --- | --- | --- |
| 0 | A, B, C, D | Yes | Yes |
| 1 | E, F, G, H | Yes | Yes |
| 2 | I | No | No |

Nine divided by four gives two complete blocks and one remaining token. Therefore, the target length of `req.block_hashes` is two.

Here, “complete” means that the token sequence contains all four token IDs for a block. It does not mean the model has already computed their KV data. This method only creates identifiers from token IDs.

### Find the next block to hash

The loop runs while the number of existing hashes is smaller than the number of complete token blocks.

Inside the loop, `i = len(req.block_hashes)` gives the index of the next logical block that needs a hash. If the list is empty, `i` is zero. After one hash is appended, `i` becomes one on the next iteration.

The slice `req.token_ids[i * bs:(i + 1) * bs]` selects that block's tokens. Python excludes the slice's end position, so with `bs = 4`:

- For `i = 0`, the slice is `[0:4]`: A, B, C, D.
- For `i = 1`, the slice is `[4:8]`: E, F, G, H.

These are logical positions within the request. Logical block 0 here is the first token block; it is not the reserved physical block 0 in the GPU block pool.

### Include the parent hash

The parent is the last hash already in `req.block_hashes`. For the first block, the list is empty, so the parent is `None`.

Using symbolic names for the resulting hashes, the first two iterations produce:

```python
H0 = hash((None, (A, B, C, D)))
H1 = hash((H0, (E, F, G, H)))
```

The inner token sequence is converted to a tuple because a Python list cannot be hashed, while a tuple of integer token IDs can. The outer tuple combines the parent hash and those tokens into the single input passed to `hash()`.

Including `H0` in the input for `H1` ties the second block's identity to the earlier prefix. If the earlier tokens change, the second block's hash input changes even if E, F, G, and H stay the same. The same principle continues through later blocks, subject to the collision limitation discussed earlier.

After the second iteration, `req.block_hashes` contains `[H0, H1]`. Its length is now two, matching `9 // 4`, so the loop stops. No hash is created for the incomplete block containing I.

### Add hashes as the token sequence grows

Suppose the request later appends J, K, and L. It now has twelve tokens, so it needs three hashes. The first two already exist; the method starts at `i = 2` and adds only:

```python
H2 = hash((H1, (I, J, K, L)))
```

| Token count when called | Complete blocks | Existing hashes | Hashes appended |
| --- | --- | --- | --- |
| 9 | 2 | 0 | H0, H1 |
| 9, called again | 2 | 2 | None |
| 10 | 2 | 2 | None |
| 11 | 2 | 2 | None |
| 12 | 3 | 2 | H2 |

This works because the method assumes the token prefix covered by existing hashes stays unchanged and `req.num_tokens` agrees with the token sequence. It only appends hashes; it does not repair old hashes if earlier tokens are edited.

### What changes after the call?

Only `req.block_hashes` is extended. The method does not allocate a physical block, compute KV tensors, or register entries in `self.block_hash` or `self.hash_to_block`.

The request's hash list identifies its token prefixes. The manager's lookup structures track where cached KV data for those prefixes is stored. Creating a hash alone does not establish that the KV data is available for reuse.

The method also does not check `enable_caching` itself. As we will see next, `find_cached_prefix` checks that flag before calling it.

### Diagram 6: building and extending the hash chain

![Complete token blocks produce H0 and H1; extending the partial block to four tokens adds H2 without recomputing earlier hashes.](assets/diagram-06-hash-chain.png)

The first two complete blocks produce H0 and H1, while `[I]` has no hash yet. After J, K, and L are appended, the new block combines with the stored parent H1 to produce `H2 = hash((H1, (I, J, K, L)))`. H0 and H1 remain unchanged, and the request's hash list grows from `[H0, H1]` to `[H0, H1, H2]`.



## Finding a cached prefix

Method: `find_cached_prefix` · [Source](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L309)

Now that the request has a way to identify its token blocks, we can look for their KV data in the cache.

`find_cached_prefix` returns the physical block IDs for the longest cached prefix allowed by this method's reuse limit. It walks from the beginning of the request and stops at the first missing block.

Here is the method with the comments condensed. The logic is unchanged:

```python
def find_cached_prefix(self, req: Request):
    """Find a contiguous cached prefix, leaving tokens to run for logits."""
    if not self.enable_caching:
        return []

    self._update_hashes(req)

    # Reuse only full blocks, and leave at least one token to process.
    max_blocks = (req.num_tokens - 1) // self.block_size

    hits = []
    for h in req.block_hashes[:max_blocks]:
        b = self.hash_to_block.get(h)
        if b is None:
            break
        hits.append(b)

    return hits
```

### Check whether prefix caching is enabled

If `enable_caching` is false, the method immediately returns an empty list. It does not update the request's hashes or look up any blocks.

An empty result means this lookup provides no prefix blocks to reuse. The request can still use a KV cache as it runs; prefix caching is the reuse of work already available from matching prefixes.

### Refresh the request's hashes

If caching is enabled, the method calls `_update_hashes(req)`. This ensures the request has hashes for every complete block of its current token sequence, preserving hashes already calculated.

For the ten-token example in the comments, with a block size of four, the complete token blocks are A–D and E–H. I and J form an incomplete final block. Using symbolic names again, the hashes are:

```python
H0 = hash((None, (A, B, C, D)))
H1 = hash((H0, (E, F, G, H)))
```

Another request with the same token prefix produces the same hash inputs. Within the same running process, those inputs produce the same hash values regardless of which request they came from. That is what allows the new request to look up an earlier request's cached blocks. This does not require the requests themselves to be identical after the shared prefix.

### Leave tokens to compute the next-token logits

Why use `(req.num_tokens - 1) // self.block_size` instead of simply counting all complete blocks?

The KV cache in this design stores keys and values. It does not store the final hidden state or the logits needed to choose the next output token.

To produce those logits, the model must process the last prompt token through its layers and then the language-model head. If every prompt token were treated as already processed through prefix reuse, there would be no remaining token to run and obtain those logits in this execution path.

For a nonempty request, subtracting one before dividing limits reuse to complete blocks that leave at least one token for model computation.

Take an eight-token prompt with a block size of four. Both token blocks may be cached, but the reuse limit is:

```python
max_blocks = (8 - 1) // 4  # 1
```

Only the first block is reused. The remaining four tokens, including the last prompt token, are processed by the model.

Because reuse happens in whole blocks, this rule can leave more than one token to compute. Assuming all eligible blocks are cache hits:

| Prompt tokens | Block size | Maximum blocks reused | Tokens reused | Tokens left to process |
| --- | --- | --- | --- | --- |
| 1 | 4 | 0 | 0 | 1 |
| 8 | 4 | 1 | 4 | 4 |
| 9 | 4 | 2 | 8 | 1 |
| 10 | 4 | 2 | 8 | 2 |
| 12 | 4 | 2 | 8 | 4 |

The last prompt token is always among those left to process. If a prefix miss happens earlier, more tokens remain.

### Look up blocks from left to right

The slice `req.block_hashes[:max_blocks]` selects only the hashes eligible for reuse. The loop visits them in request order.

For each hash, `self.hash_to_block.get(h)` returns the physical block ID if an entry exists. If there is no entry, it returns `None`, and the loop stops. Otherwise, the block ID is appended to `hits`.

For example, suppose three hashes are eligible and the cache contains:

| Prefix hash | Lookup result | Action |
| --- | --- | --- |
| H0 | Physical block 17 | Append 17. |
| H1 | Physical block 42 | Append 42. |
| H2 | Missing | Stop. |

The method returns `[17, 42]`. The order matches the request's logical block order, even though physical block IDs do not have to be consecutive.

### Why stop at the first miss?

This method finds a **contiguous cached prefix**. Once a block is missing, the uninterrupted cached portion from the start of the request ends. The code does not keep searching for later hits or return a mixture of cached blocks and gaps.

A miss means the cache lookup has no KV block for that hash. It does not mean the request's earlier tokens changed. The parent-hash chain already identifies the expected token context; stopping at the miss is how this method keeps its result limited to a prefix.

### What does the result tell us?

If the result contains two blocks and the block size is four, it identifies a cached prefix covering eight tokens. The remaining token count is `req.num_tokens - len(hits) * self.block_size`. Later scheduling and allocation code can use that information to decide what still needs computation.

The lookup itself does not increment reference counts, remove blocks from `free`, or reserve them for this request. Its only possible update to request state is extending `req.block_hashes` through `_update_hashes`. The later code must claim the returned blocks before using them.

### Diagram 7: prefix hits and the logits limit

*Illustration pending: contiguous prefix hits and the final-token logits limit.*

## Allocating blocks

Method: `allocate` · [Source](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L366)

Finding a cached prefix only gives us block IDs. `allocate` attaches those blocks to the request and assigns any additional blocks needed for the next step.

Its inputs are the request, `num_new_tokens` to process in this step, and the sequence of `cached` block IDs to attach. `cached` defaults to the empty tuple `()`, so calls without prefix hits can omit it. The return value is `True` when allocation succeeds and `False` when the capacity check fails.

The request's `block_ids` is its page table: an ordered list connecting logical token blocks to physical KV blocks. For a block size of four, `block_ids[0]` holds tokens 0–3, `block_ids[1]` holds tokens 4–7, and so on. The physical IDs themselves can be in any order.

We will divide this method into five parts: calculate capacity, check the free list, claim cached blocks, assign blocks for new data, and register full blocks.

### Start with one example

Assume a block size of four and a request with fourteen known token IDs, represented by A through N. No tokens have been computed for this request yet, and its page table is empty.

The prefix lookup returns two blocks containing A–D and E–H. We schedule computation for the remaining six tokens, I–N.

| Item | Value before allocation |
| --- | --- |
| `block_size` | 4 |
| `req.num_tokens` | 14 |
| `req.num_computed_tokens` | 0 |
| `req.block_ids` | `[]` |
| `cached` | `[17, 42]`, in prefix order |
| `num_new_tokens` | 6 |
| `free`, front to back | `[17, 23, 31, 44]` |
| `ref_cnt[17]` | 0: cached, but currently unused |
| `ref_cnt[42]` | 1: another request is using it |

Block 17 is both cached and free. Block 42 is cached and in use. Both can be prefix hits, but only block 17 needs to be removed from the free list.

In this example, `cached` is attached once, before the request has any blocks. Later calls that extend this request pass no cached prefix again. The method appends IDs without deduplicating them, so the caller must avoid passing blocks already attached to the request or counting the same cached tokens twice.

### Part 1: calculate how many blocks are needed

```python
total = (
    req.num_computed_tokens
    + len(cached) * self.block_size
    + num_new_tokens
)

total_needed_blocks = -(-total // self.block_size)
need = total_needed_blocks - len(req.block_ids) - len(cached)
```

`total` is the number of token positions the page table needs to cover after the scheduled step. It combines already-computed positions, positions supplied by the cached blocks being attached, and positions that will be processed in this step.

`num_new_tokens` means newly scheduled computation here. Those positions may be part of a known prompt; they are not necessarily newly generated output tokens.

For our example:

```text
total = 0 + 2 * 4 + 6 = 14 token positions
total_needed_blocks = ceil(14 / 4) = 4 blocks
need = 4 - 0 - 2 = 2 additional blocks
```

The expression `-(-total // self.block_size)` performs ceiling division using integers. Fourteen positions need four blocks because three blocks only cover twelve positions.

We subtract `len(req.block_ids)` because those blocks are already assigned. We also subtract `len(cached)` because the cached blocks will supply storage without needing additional blocks for their tokens.

`need` is therefore the number of blocks to take from the free list for new KV data. These blocks do not have to be blank; they may contain an older cached prefix that will be evicted.

On a later decode step, `need` can be zero because a partially filled block still has space:

| Already computed | Cached blocks being attached | New positions | Existing blocks | Required blocks | `need` |
| --- | --- | --- | --- | --- | --- |
| 14 | 0 | 1 | 4 | 4 | 0 |
| 16 | 0 | 1 | 4 | 5 | 1 |

With fourteen positions computed, the fifteenth fits in the existing fourth block. With sixteen computed, the seventeenth needs a fifth block.

### Part 2: count cached blocks that will leave the free list

```python
reclaim = sum(
    1 for b in cached if self.ref_cnt[b] == 0
)

if need + reclaim > self.num_free:
    return False
```

Checking only `need` would miss something important. A cached block with zero references is still in `free`. Claiming it for this request removes one entry from that same pool.

Here, `reclaim` counts those cached blocks. It does not count blocks whose data will be discarded; their existing KV data is exactly what we want to keep.

In our example, block 17 has zero references and block 42 has one, so `reclaim = 1`. We need two blocks for new data and one free-list entry for the cached block we are claiming:

```text
need + reclaim = 2 + 1 = 3
num_free = 4
```

The allocation can proceed. It will remove three entries from `free` in total.

If the free list contained only `[17, 23]`, the check would fail: three entries are needed, but only two are available. The method returns `False` before changing references, the page table, or cache mappings. It does not preempt a request itself; the scheduler decides how to respond.

This arithmetic assumes a consistent request state as it advances through tokens. Existing storage and newly attached cached blocks must be accounted for without overlap.

### Part 3 — first loop: claim the cached blocks

```python
for b in cached:
    if self.ref_cnt[b] == 0:
        del self.free[b]

    self.ref_cnt[b] += 1
    req.block_ids.append(b)
```

This loop attaches existing KV data to the request. It does not copy the data into another block.

For each cached block, there are two cases:

| Cached block state | Free-list action | Reference-count action |
| --- | --- | --- |
| Unused: `ref_cnt[b] == 0` | Remove that specific ID from `free`. | Increase from 0 to 1. |
| Already in use: `ref_cnt[b] > 0` | None; it is already outside `free`. | Add this request's reference. |

Now follow our two iterations:

| Iteration | Block | Reference count | Free list afterward | Request page table afterward |
| --- | --- | --- | --- | --- |
| 1 | 17 | 0 → 1 | `[23, 31, 44]` | `[17]` |
| 2 | 42 | 1 → 2 | `[23, 31, 44]` | `[17, 42]` |

Removing block 17 by ID is why the free list needs efficient removal from the middle as well as the front. A prefix hit must claim the matching block, wherever it happens to sit in the queue.

After this loop, both blocks remain cached, and both are actively referenced. Being removed from the free list does not remove their prefix hashes or invalidate their KV data. Block 42 is now shared by two requests.

Claiming these blocks first also prevents the next loop from selecting block 17 for unrelated new data.

### Part 4 — second loop: assign blocks for new KV data

```python
for _ in range(max(need, 0)):
    b, _ = self.free.popitem(last=False)

    if self.block_hash[b] is not None:
        self.hash_to_block.pop(self.block_hash[b], None)
        self.block_hash[b] = None

    self.ref_cnt[b] = 1
    req.block_ids.append(b)
```

The first loop reused known KV contents. This loop assigns physical storage for the tokens we still need to compute.

`range(max(need, 0))` runs once per additional block needed, or not at all if existing storage is sufficient. Each iteration takes the front entry with `popitem(last=False)`, following the free-list order.

Suppose block 23 still represents an older prefix P, while block 31 has no registered hash:

| Iteration | Selected block | Old metadata | Action | Request page table afterward |
| --- | --- | --- | --- | --- |
| 1 | 23 | `block_hash[23] = P`, `hash_to_block[P] = 23` | Remove P's mapping and clear block 23's hash. Set its reference count to 1. | `[17, 42, 23]` |
| 2 | 31 | `block_hash[31] = None` | No old mapping to remove. Set its reference count to 1. | `[17, 42, 23, 31]` |

The free list is now `[44]`.

Why clear the old hash? The block is about to hold different KV data. Leaving P mapped to block 23 would let a future lookup mistake that new data for P's old prefix. The old entry must stop being searchable when its storage is reassigned.

The `.pop(..., None)` call tolerates an already-absent mapping. Clearing the metadata does not zero the GPU memory or compute the new values; the model execution will write the new KV data.

The two loops therefore have different purposes:

| Loop | What happens to the existing KV contents? | What happens to ownership? |
| --- | --- | --- |
| Claim `cached` | Keep them for reuse. | Add a reference and attach the block. |
| Allocate `need` blocks | Discard any old prefix identity so storage can be overwritten. | Assign the block with one reference and attach it. |

### Part 5: register full blocks and return

```python
self._register_full_blocks(req, total)
return True
```

After both loops, the page table covers the scheduled token positions. The method calls `_register_full_blocks` to associate eligible full blocks with their prefix hashes, then reports success.

Our example ends with this layout:

| Logical block | Physical block | Token positions | Source | Reference count after allocation |
| --- | --- | --- | --- | --- |
| 0 | 17 | A–D | Reused prefix | 1 |
| 1 | 42 | E–H | Reused, shared prefix | 2 |
| 2 | 23 | I–L | Storage assigned for computation | 1 |
| 3 | 31 | M–N, with two spare positions | Storage assigned for computation | 1 |

`allocate` does not advance `req.num_computed_tokens` or run the model. It prepares storage and metadata for the step. The computation and progress updates belong to the surrounding execution path.

### Diagram 8: the two allocation loops

![The request claims cached blocks 17 and 42, then receives blocks 23 and 31 for new computation, leaving block 44 free.](assets/diagram-08-allocation-loops.png)

The first loop attaches cached blocks 17 and 42. Only 17 leaves `free`, because 42 is already in use; their reference counts become one and two. The second loop takes 23 and 31 from the front of `free`, removes block 23's old prefix P, and extends the page table to `[17, 42, 23, 31]`. Only 44 remains free.

The input sequence `cached` remains `[17, 42]` throughout. Both reused blocks retain their prefix identities, H0 and H1; P is the unrelated old prefix being evicted from block 23.



## Registering full blocks

Method: `_register_full_blocks` · [Source](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L456)

Once storage has been assigned, this helper records prefix hashes for full blocks within the covered, known token range.

```python
def _register_full_blocks(self, req: Request, num_tokens: int):
    if not self.enable_caching:
        return

    self._update_hashes(req)

    for i in range(min(num_tokens, req.num_tokens) // self.block_size):
        b, h = req.block_ids[i], req.block_hashes[i]

        if self.block_hash[b] is None:
            self.block_hash[b] = h
            self.hash_to_block.setdefault(h, b)
```

### Limit registration to covered, known, complete blocks

If caching is disabled, the method returns immediately. Otherwise, `_update_hashes` ensures hashes exist for the request's complete token blocks.

The loop bound combines three limits:

- `num_tokens` is the token range being covered; `allocate` passes its calculated `total`.
- `req.num_tokens` limits registration to tokens whose IDs the request knows.
- Integer division by `block_size` excludes a partial final block.

For our example, `min(14, 14) // 4` is three. The loop visits logical blocks 0, 1, and 2. It skips logical block 3 because M and N do not fill it.

The `min` also matters for chunked prefill. If a request already knows twenty prompt token IDs but this step only covers the first six positions, `min(6, 20) // 4` is one. The helper registers only the first complete block, not every full block whose hash can be calculated from the longer prompt.

### Pair each physical block with its logical prefix hash

At each logical index `i`, the code reads `req.block_ids[i]` and `req.block_hashes[i]`. One identifies the physical storage; the other identifies the token prefix that storage represents.

Using H0, H1, and H2 for our example:

| Logical index | Physical block `b` | Hash `h` | Result |
| --- | --- | --- | --- |
| 0 | 17 | H0 | Already has a hash; leave unchanged. |
| 1 | 42 | H1 | Already has a hash; leave unchanged. |
| 2 | 23 | H2 | Set `block_hash[23] = H2`; add H2's lookup entry if absent. |

Block 31 remains unregistered because its logical token block is incomplete.

Unlike `_update_hashes`, which appends only missing hashes, this loop revisits every complete block in the covered range. The `if self.block_hash[b] is None` guard makes it skip blocks that already have a registered identity. It assumes an existing identity is correct for that position; it does not compare and repair mismatched hashes.

### Why use `setdefault`?

`self.hash_to_block.setdefault(h, b)` inserts `h → b` only if h is not already a key. If another physical block is already indexed for that prefix, the existing lookup stays in place.

For example, if H2 already maps to block 55, registering block 23 sets `block_hash[23] = H2` but leaves `hash_to_block[H2] = 55`. This keeps one searchable representative for that hash, even though more than one physical block may carry the same prefix identity.

That detail also exposes an assumption in the eviction code above: it removes a block's old hash from `hash_to_block` without checking which physical block the entry currently points to. In this example, later reassigning block 23 would remove H2's mapping to block 55 as well. The remaining code needs to establish whether duplicate registrations can occur in practice; if they can, eviction should check that the mapping points to the block being reassigned before deleting it.

### Registration is separate from computing KV data

This helper records metadata. It does not run attention or write keys and values.

Because `allocate` calls it while preparing a step, a newly registered block may still be waiting for its model computation. For example, block 23 has H2's identity after allocation, but the code shown here has not computed I–L's KV data.

The scheduler and model execution must ensure that another request does not consume a newly registered block before its KV data is ready. We will check that ordering when the execution code is available. A hash entry alone is not evidence that the model work has finished.

### Diagram 9: physical blocks and prefix lookup

![Physical blocks 17, 42, and 23 are indexed by H0, H1, and H2; partial block 31 has no registered prefix hash.](assets/diagram-09-prefix-lookup.png)

The page table pairs each logical block with its physical location. Assuming H2 was not already indexed, the lookup contains H0 → 17, H1 → 42, and H2 → 23. Block 31 is partial and remains outside the prefix lookup. Registering H2 records the destination for this step's KV computation; the model still needs to produce block 23's new contents.

## Releasing a request

Method: `free_request` · [Source](https://github.com/sudhirpol522/small-vLLM/blob/62bfcedf01abcaa3df3d1905fd6028558e35dead/serve.py#L482)

When a request finishes, its blocks should become available to other requests unless they are still shared. `free_request` releases this request's references and clears its page table.

The same operation can release ownership during preemption. Deciding when to call it and how to reschedule the request belongs to the scheduler.

```python
def free_request(self, req: Request):
    for b in reversed(req.block_ids):
        self.ref_cnt[b] -= 1
        if self.ref_cnt[b] == 0:
            self.free[b] = None
    req.block_ids = []
```

### Continue the allocation example

Use the page table from the earlier example: `[17, 42, 23, 31]`. Assume the scheduled KV computation has completed and the request still has these four blocks when it is released.

| Physical block | Tokens | Reference count before release | Registered prefix hash |
| --- | --- | --- | --- |
| 17 | A–D | 1 | H0 |
| 42 | E–H | 2: this request and another request | H1 |
| 23 | I–L | 1 | H2 |
| 31 | M–N, with two spare positions | 1 | `None`: incomplete block |

The free list contains `[44]`. None of the request's four blocks is currently free because each has at least one active reference.

### Part 1: visit the blocks in reverse order

```python
for b in reversed(req.block_ids):
```

The page table is ordered from the beginning of the token sequence to its end. `reversed` visits the tail first, so the release order is 31, 23, 42, then 17.

The physical IDs are not being sorted. The loop reverses their logical order within this particular request.

### Part 2: remove this request's reference

```python
self.ref_cnt[b] -= 1
```

Each block loses one reference because this request no longer owns it. The method does not set every count directly to zero: another request may still depend on the same cached block.

For example, block 42 goes from two references to one. That remaining reference protects it from reassignment even though the request we are releasing is finished.

The code assumes that every ID in the page table represents a valid reference held by this request.

### Part 3: append only unreferenced blocks to the free list

```python
if self.ref_cnt[b] == 0:
    self.free[b] = None
```

When the count reaches zero, no request is using the block. Assigning its previously absent ID into the `OrderedDict` appends it at the back. The value `None` is just a placeholder; the key is the physical block ID.

Follow the loop one iteration at a time:

| Iteration | Block released | Reference count | Free-list action | Free list afterward, front to back |
| --- | --- | --- | --- | --- |
| 1 | 31 | 1 → 0 | Append 31. | `[44, 31]` |
| 2 | 23 | 1 → 0 | Append 23. | `[44, 31, 23]` |
| 3 | 42 | 2 → 1 | Keep it in use; do not append. | `[44, 31, 23]` |
| 4 | 17 | 1 → 0 | Append 17. | `[44, 31, 23, 17]` |

Four references were released, but only three blocks became free. Shared block 42 stays outside the free list until its final user releases it.

### Why release the tail first?

New allocations take blocks from the front of `free`, while released blocks are appended at the back. Releasing in reverse logical order puts later token blocks before earlier prefix blocks among the blocks returned by this call.

In our result, block 31 is ahead of block 23, and block 23 is ahead of block 17. If no other operation changes the queue, allocations for new data will select 44, then 31, then 23, then 17.

This ordering favors keeping earlier prefix blocks longer. A later request can only reuse the contiguous cached prefix from the beginning, so preserving an early block can help preserve a useful run of prefix hits. It is a preference within this release operation, not a guarantee that an early block will survive memory pressure.

### Releasing ownership keeps the cached contents

Notice that this method never clears `block_hash` or removes entries from `hash_to_block`. It also does not erase KV tensors.

After release, H0 can still point to block 17 and H2 can still point to block 23, even though those blocks are free. H1 still points to block 42, which remains in use by the other request. Block 31 becomes free without a prefix entry because its token block is incomplete.

Suppose another request now has ten tokens and shares the same first eight tokens. A prefix lookup can still find blocks 17 and 42, leaving the last two tokens to process. The cached-block loop in `allocate` removes 17 from the free list and changes its count from zero to one. Block 42 is already in use, so it stays outside `free` and its count rises from one to two.

That is the connection between releasing a request and retaining prefix reuse: **free means available for reassignment; it does not mean the old cached data has already been discarded**. If block 17 or 23 is selected for different data first, the allocation loop removes its old prefix mapping at that point.

### Part 4: clear this request's page table

```python
req.block_ids = []
```

After the loop, the request no longer owns any blocks. Clearing its list also means a second call, before any new blocks are attached, has no references to release again.

The comment about clearing everything should be read as clearing block ownership. This method does not reset `req.num_computed_tokens`, erase token IDs, clear `req.block_hashes`, or change request status. If the request is being preempted and will run again, the surrounding code must update its progress to reflect the KV data it will need to recover or recompute.

The final state is `req.block_ids = []`, `free = [44, 31, 23, 17]`, and one remaining reference to shared block 42. The method visits each attached block once, so its work grows with the number of blocks the request held.

### Diagram 10: releasing ownership while retaining prefixes

![Releasing the request leaves shared block 42 in use and returns blocks 31, 23, and 17 to the free queue while retaining valid prefix mappings.](assets/diagram-10-release-ownership.png)

Releasing in reverse page-table order returns 31, then 23, then 17, producing `free = [44, 31, 23, 17]`. Block 42's count falls from two to one, so it stays in use by the other request. This request's page table becomes empty, but H0 → 17, H1 → 42, and H2 → 23 remain available for prefix lookup. Block 31 is free without a registered prefix.

Next: the scheduler chapter is planned. Definition locations are available in the [source index](api-index.md).
