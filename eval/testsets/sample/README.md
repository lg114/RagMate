# 样本评测集（重建版）

本目录是 2026-11-02 重建的精选小测试集：原始 10 题测试集与语料只在
作者本地、已丢失，按原实验设计（`RETRIEVAL_EXPERIMENTS.md` 阶段 0 与
`eval/reports/baseline/retrieval-stage0.md`）重做并扩容。参考答案全部手写，
逐条可从语料原文核对，没有用 LLM 生成。

## 内容

| 文件 | 说明 |
|---|---|
| `testset.json` | 50 道问答，格式与 `ragmate-eval evaluate` 的输入一致（`user_input` + `reference`） |
| `corpus/*.md` | 8 篇中文技术文档，约 24,000 字：Docker Compose 部署、PostgreSQL 索引优化、Redis 缓存策略、Python asyncio、HTTP 缓存与 CDN、JWT 与 OAuth 2.0、Linux 进程与信号、Git 内部原理 |

题目覆盖：单文档事实问答、跨概念对比、机制问答，以及 2 道语料未覆盖
问题（Kubernetes HPA、Terraform remote state，考察模型不 hallucinating、
如实回答无法回答）。每篇文档约 5–6 道题，题目均匀分布，避免单篇语料
主导评测结果。

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

- 样本量中等（50 题），结论只适用于回归对比与配置间比较，不外推为
  普遍效果；再往 100 题扩时应继续扩语料，避免题目之间互相提示答案；
- 语料与题目均为人工编写的“干净”技术文档，行文规范、术语集中，
  检索难度低于真实杂乱文档，分数会比真实场景偏高，只宜纵向对比。
