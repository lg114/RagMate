# Python asyncio 并发模型笔记

本文记录 RagMate 后端（Python 3.12）里 asyncio 的使用约定，目标是
I/O 密集型任务的并发，不涉及多进程与 CPU 密集型计算。

## 事件循环：单线程内调度协程

asyncio 的事件循环在**单线程**内调度协程：同一时刻只有一个协程在执行，
协程在 `await` 处主动让出控制权，循环再调度下一个就绪的协程。协程之间
没有真正的并行，也就没有数据竞争，但一个协程里长时间不 `await` 的
CPU 计算会堵住整个循环。

启动入口用 `asyncio.run()`：它创建新的事件循环、运行传入的协程、结束
时关闭循环。不要在已有运行中循环的线程里再调 `asyncio.run()`，会报
`RuntimeError: asyncio.run() cannot be called from a running event loop`；
那种场景改用 `await` 直接调用。

## gather：并发运行，异常默认传播

`asyncio.gather()` 并发运行多个协程并收集结果：

```python
results = await asyncio.gather(fetch_a(), fetch_b(), fetch_c())
```

默认行为（`return_exceptions=False`）：其中一个协程抛异常时，异常会
传播给调用者，但**其他协程继续运行**，不会被取消；`gather()` 返回的
结果里只有先完成的那个异常。想拿到全部结果（包括异常对象本身）时，
传 `return_exceptions=True`，此时返回值列表里异常以异常对象形式出现，
调用方需要逐个检查类型。

`gather()` 适合彼此独立的任务：三个 HTTP 请求互不依赖，一个失败不该
取消另外两个。

## TaskGroup：出错时取消同组任务

Python 3.11 引入的 `asyncio.TaskGroup` 是更严格的任务组：

```python
async with asyncio.TaskGroup() as tg:
    tg.create_task(fetch_a())
    tg.create_task(fetch_b())
```

TaskGroup 的语义与 `gather()` 不同：组内**任何一个任务抛异常**，
TaskGroup 会取消组内其余所有任务，并等待它们全部结束，退出 `async with`
块时把第一个异常重新抛出。任务组保证“要么全部完成，要么整体失败”，
适合有依赖关系的任务：一组入库步骤里某一步失败，继续跑剩下的步骤已
经没有意义。

退出 `async with` 块时，TaskGroup 会等待组内所有任务结束，调用方不用
再手动 `await` 每个任务。

## Semaphore：限制并发数

向外部 API 发请求时，用信号量限制同时在途的请求数，避免把对方打垮：

```python
sem = asyncio.Semaphore(10)

async def fetch_one(url):
    async with sem:
        return await http_get(url)
```

`asyncio.Semaphore(10)` 表示最多 10 个协程同时持有许可，第 11 个协程
在 `async with sem` 处等待，直到有协程释放许可。信号量只限制并发数，
不限制速率：10 个请求同时发出后立刻又有 10 个补上，需要限速时再配合
令牌桶或固定窗口。

## 超时：asyncio.timeout

Python 3.11 的 `asyncio.timeout()` 替代了旧的 `wait_for` 写法：

```python
async with asyncio.timeout(5):
    result = await fetch_slow()
```

超过 5 秒未完成时，当前任务被取消并抛出 `TimeoutError`。`timeout()`
只取消它包裹的那个任务，不影响同事件循环里的其他任务。超时时间从进入
`async with` 块开始算，块内多次 `await` 共享同一个截止时间。

## 常见误区

- 在协程里调用同步阻塞函数（如 `time.sleep()`、`requests.get()`）会堵住
  事件循环，整个应用的并发全部停摆；阻塞调用改用 `asyncio.to_thread`
  扔到线程池，或换异步库。
- `asyncio.create_task()` 创建的任务如果没人持有引用，可能在运行中途
  被垃圾回收，表现为任务静默消失；需要长期运行的任务要把 Task 对象
  保存到集合里。
- 异常不要吞：在 `gather(return_exceptions=True)` 拿到结果后必须检查每个
  元素是不是异常对象，否则失败会被当成正常结果继续往下传。

## 队列：生产者-消费者模式

`asyncio.Queue` 是协程间传递任务的标准结构，`maxsize` 限制队列容量：

```python
queue: asyncio.Queue = asyncio.Queue(maxsize=100)

async def producer():
    for item in items:
        await queue.put(item)   # 队列满时等待，直到有空位

async def consumer():
    while True:
        item = await queue.get()
        try:
            await process(item)
        finally:
            queue.task_done()
```

`put()` 在队列满时挂起生产者，天然形成背压：生产速度超过消费速度时，
生产者自动降速，不会无限制堆积内存。`maxsize=0` 表示无界队列，生产者
永不等待，内存风险自担，本文约定一律显式设 `maxsize`。

`queue.join()` 等待所有已 `put` 的任务都被 `task_done()` 标记完成，
配合 `task_done()` 的 `finally` 保证：处理函数抛异常时计数器也正确
扣减，否则 `join()` 永远等下去。消费者退出条件用哨兵值：生产者结束
后往队列里放一个 `None`，消费者读到 `None` 即退出循环。

## 阻塞调用：扔到线程池

事件循环里必须避免同步阻塞调用，确实躲不开时用 `asyncio.to_thread`
扔到线程池：

```python
result = await asyncio.to_thread(blocking_read_file, path)
```

`to_thread` 在默认线程池里运行阻塞函数，事件循环继续调度其他协程。
线程池默认大小为 `min(32, os.cpu_count() + 4)`，阻塞任务长期占满线程
池时新任务排队等待，排队本身不会报错，但延迟会上升。`to_thread` 只
解决“不堵循环”，不解决“阻塞调用本身很慢”，真正高频的阻塞操作要换
异步实现。

旧写法 `loop.run_in_executor(None, func)` 与 `to_thread` 等价，新代码
统一用 `to_thread`。注意 `to_thread` 里的函数如果抛异常，异常会传播
到 `await` 处，与普通协程的异常处理方式一致。

## 取消与清理：finally 与 shield

取消一个任务时，事件循环在任务当前 `await` 的位置抛出
`CancelledError`，任务里的 `finally` 块仍会执行，适合放资源清理：

```python
async def work():
    conn = await acquire()
    try:
        await do_something(conn)
    finally:
        await conn.close()
```

`finally` 里不要再做长时间的 `await`：取消后 `finally` 里的 `await`
也可能被再次取消，清理逻辑本身要短小、且对重复取消安全。清理时需要
发起新的异步操作（例如发一条取消通知）时，用 `asyncio.shield()` 包
住那次调用，让它不受外层取消影响：

```python
await asyncio.shield(send_cancel_notice())
```

`shield()` 只保护被包裹的那一次 await，不保护整个任务；任务本身在
`shield()` 返回后仍然是已取消状态，`CancelledError` 会继续传播。
不要用 `except CancelledError: pass` 把取消吞掉后假装任务正常完成，
调用方（例如 TaskGroup）靠这个异常判断任务是被取消还是成功，吞掉后
任务组语义会被破坏。
