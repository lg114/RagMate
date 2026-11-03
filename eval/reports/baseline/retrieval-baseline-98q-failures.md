# 98q 检索失败清单（相对 `retrieval-baseline-98q`）

只列「至少一臂 MRR&lt;1 或 recall@5&lt;1」的可答题。unanswerable 不计入。  
三遍确定性成立；下列 rank 来自第一遍 checkpoint。

## 总览

| 机制 | 题号 | 一句话 |
|---|---|---|
| 多证据分散 + 近义干扰块抢位 | **67** | 三块重启策略拆在 web/db/redis 三节；worker 块因「unless-stopped」更显眼压过 gold |
| 跨文档第二证据被同主题文档淹没 | **91** | Compose 的 AOF 短块在 sparse/fused 掉出 top5；rerank 拉回 #2 |
| 改写缺术语，sparse 失手 | 55, 53 | 融合/ dense 能救；说明 hybrid 对 paraphrase 有用 |
| 改写后 dense 被邻段抢走 | 50, 54 | sparse 或融合拉回 #1 |
| 单路 rank2，融合或 rerank 修 | 0,10,38,41,44,48,58,75 | 非系统性；多在 factoid 的 1↔2 |

`recall@20=1.0`：缺的 gold 都已进候选池，**加大 over_fetch 解决不了剩下的缺口**。

---

## 题 67（最硬）— 多证据 + 干扰块

**问：** Compose 里 db 与 redis 的重启策略分别是什么？和 web/worker 有何不同？  
**Gold：** `(01,2)` web→unless-stopped · `(01,4)` db→always · `(01,5)` redis→always

| 阶段 | gold ranks | top1 干扰 |
|---|---|---|
| dense | 11, 14, **4** | worker 块（unless-stopped 写在段首） |
| sparse | 4, 6, 7 | 同上 |
| fused | 6, 8, **5** | worker / 四服务总览 / 网络段 |
| reranked | **2, 4, 6** | worker 仍 #1；redis gold 卡在 #6 |

**机制：**

1. 答案要凑齐三块，但「重启策略」在 web/db 长段里只是附带一句，在 worker 短段里却是主题句 → 检索偏爱 worker。
2. redis 的 `always` 在短块 (5) 里写得很清楚，dense 能拿到 #4，但 rerank 时被同文档其它段挤到 #6。
3. 若生产 `FINAL_CONTEXT_K=5`，rerank 后仍会丢掉 redis 那块；`k=15` 能兜住。这是**排序/截断问题，不是召回问题**。

**对后续实验的含义：** 扫候选池无用。更相关的是：(a) 标题路径/按服务切块，让「重启策略」不要埋在端口映射长段里；(b) 多跳题看 recall@k 与最终 k，不要只看 hit@5；(c) 查询改写拆成「web/db/redis 各自重启策略」可能比加大 pool 有效。

---

## 题 91 — 跨文档，一侧被同主题淹没

**问：** 开发编排里 Redis 如何持久化？缓存文档对 appendfsync 的推荐又是什么？  
**Gold：** `(01,5)` appendonly yes · `(03,6)` appendfsync=everysec

| 阶段 | Compose 块 | 缓存块 |
|---|---|---|
| dense | 4 | 1 |
| sparse | **16** | 1 |
| fused | **9** | 1 |
| reranked | **2** | 1 |

**机制：** 问句后半「appendfsync」把 sparse/融合质量拴在 `03-redis-cache`；Compose 短块几乎不含该词，被同文档其它 Redis 段挤出 top5。候选池里仍有它（#9/#16），**rerank 能识别跨文档第二证据并拉到 #2**。

**对后续实验的含义：** 这是 rerank 真实价值的样例，不是 pool 不够。保留 fused→rerank 消融；跨文档题单独报 recall@5。

---

## Paraphrase 小结

| 题 | 谁失手 | 谁救回 |
|---|---|---|
| 55 短命/长命凭证 | sparse #3（缺 access/refresh 术语） | dense/fused/rerank #1 |
| 50 打开页面地址 | dense #2（网络段抢先） | sparse/fused #1 |
| 53 在途请求压到十个 | sparse #2 | dense/fused #1 |
| 54 「3600 秒新鲜」 | dense #2 | sparse/fused #1 |

**机制：** 改写拿掉术语 → sparse 弱；语义近邻段抢 dense → 融合补齐。与「hybrid 在 paraphrase 上有用」的消融一致。

---

## 建议的下一步（按这份清单）

1. **先改评测读法：** multi-hop 主看「全部 gold 的 recall@5/10」，不要被 hit@5=1.0 骗过。  
2. **生产截断：** 对多证据题，`FINAL_CONTEXT_K=5` 偏险（题 67）；15 更稳——可用同一 checkpoint 模拟不同最终 k，无需重跑检索。  
3. **切块实验优先于扫 pool：** 题 67 指向「服务级/标题路径切块」，与 P1 标题路径项对齐。  
4. **查询改写：** 只在题 67 类「多实体并列」上试，按题型拆开看，避免在已满分的 factoid 上空转。
