# PostgreSQL 索引与查询优化笔记

本文记录 RagMate 元数据表（PostgreSQL 15）在数据量增长后做过的索引与
查询优化，结论来自 `EXPLAIN ANALYZE` 的实测，不是经验推测。

## 先看执行计划

任何优化开始前，先跑 `EXPLAIN ANALYZE`：

```sql
EXPLAIN ANALYZE
SELECT * FROM documents WHERE status = 'active' ORDER BY uploaded_at DESC LIMIT 20;
```

关注三处：`Seq Scan`（全表扫描）、`Sort` 节点的 `Sort Method`（出现
`external merge Disk` 说明排序溢出到磁盘）、以及 `Buffers` 里的
`shared hit` 与 `shared read`（读了多少 8KB 页）。

## B-tree：默认索引，适合等值、范围与排序

B-tree 是 PostgreSQL 的默认索引类型，适合等值查询、范围查询与排序：

```sql
CREATE INDEX idx_documents_uploaded_at ON documents (uploaded_at DESC);
```

上面这条索引同时服务两类查询：按 `uploaded_at` 范围过滤，以及按
`uploaded_at DESC` 排序取最新 20 条。`ORDER BY uploaded_at DESC LIMIT 20`
可以直接走索引顺序读，不再单独排序。

复合索引遵循最左前缀原则：`(status, uploaded_at)` 能服务
`WHERE status = ? ORDER BY uploaded_at`，但单独查 `uploaded_at` 时用不上
这条索引。不要为了“以防万一”建一堆复合索引，每个索引都要占用磁盘并拖慢
写入。

## GIN：全文检索、JSONB 与数组包含查询

B-tree 处理不了的查询类型，交给 GIN（Generalized Inverted Index）：

```sql
-- 全文检索：to_tsvector 生成的词向量上建 GIN
CREATE INDEX idx_documents_fts ON documents USING GIN (to_tsvector('english', content));

-- JSONB 包含查询：metadata 里 tags 数组包含某值
CREATE INDEX idx_documents_metadata ON documents USING GIN (metadata jsonb_path_ops);
```

GIN 索引的写入代价高于 B-tree，更新频繁的表上建 GIN 要先评估写入量。
反过来，纯 B-tree 也查不了 `metadata @> '{"tags": ["rag"]}'` 这类包含查询，
会退化成全表扫描。

## 部分索引：只对满足条件的行建索引

部分索引（partial index）只对满足 `WHERE` 条件的行建立索引，索引更小、
维护成本更低：

```sql
CREATE INDEX idx_documents_active ON documents (uploaded_at DESC)
WHERE status = 'active';
```

这条索引只包含 `status = 'active'` 的行。查询条件必须与索引的 `WHERE`
谓词一致才能用上它，查 `status = 'archived'` 时优化器不会选这条索引。
适用场景是明确的热点子集：活跃文档占全表不到两成时，部分索引比全量
索引小得多，查询也更快。

## work_mem：每个排序/哈希操作可用的内存

`work_mem` 的默认值是 **4MB**，它是**每个**排序或哈希操作可用的内存，
不是全局上限：一个查询里有三个排序节点，最多可用 3 × work_mem。

排序数据量超过 `work_mem` 时，PostgreSQL 改用磁盘上的外部归并排序，
`EXPLAIN ANALYZE` 里表现为 `Sort Method: external merge Disk`。把
`work_mem` 临时调到 64MB 再跑同一条查询，排序改回内存中的
`quicksort`，耗时明显下降：

```sql
SET work_mem = '64MB';
```

不要全局调得过高：并发连接数乘以 `work_mem` 就是最坏情况下的内存占用，
64MB 配 100 个并发连接，理论峰值就是 6.4GB。生产环境按并发量留出余地，
只在会话级为重查询单独调大。

## 批量导入后先 ANALYZE

大批量导入数据后，优化器手里的统计信息还是旧的，可能继续选错执行计划。
导入完成后执行：

```sql
VACUUM ANALYZE documents;
```

`ANALYZE` 更新表的行数与列分布统计，`VACUUM` 回收死元组。只读查询为主
的场景可以只跑 `ANALYZE`，跳过 `VACUUM` 的清理开销。

## 覆盖索引与仅索引扫描

查询用到的列全部在索引里时，PostgreSQL 可以只读索引、不回表，称为
Index Only Scan：

```sql
CREATE INDEX idx_documents_covering ON documents (status, uploaded_at, filename);
```

仅索引扫描依赖可见性映射（visibility map），刚更新过的行仍需回表确认
可见性，`VACUUM` 之后效果最好。覆盖索引不是万能的：索引列越多，写入
代价越高，只给高频查询建。

## 解读 EXPLAIN：cost 与 rows

`EXPLAIN` 输出里的 `cost` 不是时间单位，而是优化器内部的代价估算，
数值越大表示优化器认为代价越高，单位与磁盘页读取的代价相关，不直接
对应毫秒。`rows` 是优化器估计的返回行数，`actual time` 与 `rows` 前的
`actual` 前缀才是实测值。

估计行数与实际行数差距超过一个数量级时，问题通常在统计信息：先跑
`ANALYZE` 更新统计，再看估计是否收敛。如果统计已更新但估计仍偏差很大，
检查查询条件里是否有高度相关的列组合（例如 `status` 与 `is_archived`
几乎一一对应），优化器按独立分布估算会系统性低估或高估。

`Buffers: shared hit=120 read=5` 表示查询读了 125 个 8KB 页，其中 120 页
命中共享缓存、5 页从磁盘读。`shared read` 长期偏高说明工作集超出
`shared_buffers`，优先考虑加索引减少扫描范围，而不是盲目调大缓存。

## 连接池：PgBouncer 事务级复用

PostgreSQL 每个连接对应一个后端进程，`max_connections` 默认 100，连接数
一多内存与上下文切换开销都上升。应用层用 PgBouncer 做连接池，池模式
选 transaction（事务级）：一个客户端连接只在事务期间占用一个服务端
连接，事务结束即归还，不同客户端连接复用同一批服务端连接。

事务级池模式的限制是：会话级特性不能用，包括会话级 advisory lock、
`SET` 的会话级配置、预编译语句（prepared statement）跨事务复用。RagMate
的查询都是短事务、无会话状态，符合事务级模式的要求。`work_mem` 这类
会话级设置在池模式下要改在连接建立时下发，不能依赖客户端 `SET` 后
长期生效。

连接池本身也要设上限：PgBouncer 的 `default_pool_size` 设为 20，超过
20 个并发事务的请求在池侧排队，排队超时（`server_idle_timeout` 之外
另有 `query_wait_timeout`）后返回错误而不是无限等待。

## 慢查询日志：先定位再优化

优化靠猜没有用，先让数据库自己报慢查询。`postgresql.conf` 里打开：

```
log_min_duration_statement = 1000
```

执行超过 1000 毫秒的语句记入日志，附带执行计划的用 `auto_explain`
模块（`shared_preload_libraries = 'auto_explain'`，重启生效），慢查询
的完整计划直接进日志，不用事后复现。

定位高频慢查询用 `pg_stat_statements` 扩展：

```sql
CREATE EXTENSION pg_stat_statements;

SELECT query, calls, mean_exec_time, rows
FROM pg_stat_statements
ORDER BY mean_exec_time DESC LIMIT 10;
```

`mean_exec_time` 是平均执行毫秒数，`calls` 是累计调用次数，两者相乘
才是总耗时：平均 5 毫秒但每天调用 200 万次的查询，总耗时超过平均
800 毫秒但每天只跑 3 次的查询。先优化总耗时最高的，再看平均耗时。

`pg_stat_statements` 的统计是累计值，改完索引后要 `SELECT
pg_stat_statements_reset()` 清零重计，否则新旧数据混在一起看不出优化
效果。清零操作影响全局统计，只能在低峰期做。
