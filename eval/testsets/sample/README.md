# 样本评测集（重建版）

本目录是 2026-11-02 重建的精选小测试集：原始 10 题测试集与语料只在
作者本地、已丢失，按原实验设计（`RETRIEVAL_EXPERIMENTS.md` 阶段 0 与
`eval/reports/baseline/retrieval-stage0.md`）重做。参考答案全部手写，
逐条可从语料原文核对，没有用 LLM 生成。

## 内容

| 文件 | 说明 |
|---|---|
| `testset.json` | 10 道问答，格式与 `ragmate-eval evaluate` 的输入一致（`user_input` + `reference`） |
| `corpus/*.md` | 4 篇中文技术文档，约 15,000 字：Docker Compose 部署、PostgreSQL 索引优化、Redis 缓存策略、Python asyncio |

题目覆盖：单文档事实问答（端口映射、健康检查参数、work_mem、部分索引、
信号量写法）、跨概念对比（TaskGroup 与 gather、GIN 与 B-tree）、机制问答
（雪崩缓解、击穿互斥锁），以及 1 道语料未覆盖问题（Kubernetes HPA，
考察模型不 hallucinating、如实回答无法回答）。

## 复现方法

1. 把 `corpus/` 下的 4 篇文档放进 `backend/documents/`（或 `DOCUMENTS_DIR`
   指向的目录），触发一次入库；
2. 在仓库根目录运行评测（需先启动 PostgreSQL / Redis / Milvus 栈，
   并配置好 LLM 密钥）：

```bash
ragmate-eval evaluate --testset eval/testsets/sample/testset.json
```

3. 结果与 `eval/reports/baseline/retrieval-stage0.md` 的阶段 0 基线对比
   时注意：基线跑的是已丢失的原语料，本集的数字是**新基线**，只用于
   回归对比（同集前后对比），不与原基线数字直接比较。

## 局限

- 样本量小（10 题），结论只适用于回归对比，不外推为普遍效果；
- 语料规模约 15,000 字，小于原实验的约 24,000 字；扩到 50–100 题时
  应先扩语料，避免题目之间互相提示答案。
