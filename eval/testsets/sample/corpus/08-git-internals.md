# Git 内部原理笔记

本文记录 Git 的对象模型与常用操作的底层语义，Git 版本 2.40+，仓库
格式为默认的 SHA-1 寻址。

## 四种对象：blob、tree、commit、tag

Git 的一切内容都是对象，存在 `.git/objects/` 下，按内容哈希寻址：

- **blob**：存文件内容，不含文件名。两个文件内容相同则对应同一个
  blob，改名不改内容不会产生新 blob；
- **tree**：存目录快照，记录该目录下每个条目的名字、模式与指向的
  blob 或子 tree 的哈希；
- **commit**：存一次提交的元数据——指向的根 tree 哈希、父 commit 哈希
  （合并提交有两个）、作者与提交者信息、时间戳与提交说明；
- **tag**：附注标签（annotated tag）是一个独立对象，指向某个 commit
  并附带标签说明；轻量标签（lightweight tag）只是分支式引用，不产生
  对象。

对象名就是内容的 **SHA-1** 哈希：内容相同则哈希相同、对象相同，内容
有任何一个字节变化则哈希完全不同。SHA-1 寻址同时保证了完整性：对象
内容被篡改后，重新计算的哈希与对象名对不上，`git fsck` 能检出。

新版 Git 支持 SHA-256 寻址（`git init --object-format=sha256`），仓库
一旦选定寻址格式不可更改，本文档的仓库统一用默认 SHA-1。

## HEAD、分支与 detached HEAD

分支只是一个指向某个 commit 的可移动引用（`.git/refs/heads/<分支名>`
文件里存一个 commit 哈希）。**HEAD** 指向当前分支（通常是
`ref: refs/heads/main`），新提交产生时，当前分支引用随之前进，HEAD
随之指向新 commit。

**detached HEAD** 指 HEAD 直接指向某个 commit 哈希、而不是分支：

```bash
git checkout 1ae121d   # HEAD 从 main 上 detached
```

这种状态下新提交不在任何分支上，HEAD 随新提交移动，但没有任何分支
引用记录它们；一旦 `checkout` 到别的分支，那些提交就只能靠 reflog
找回，久了会被 `gc` 清理。临时查看旧代码用 detached HEAD 没问题，
要长期工作先 `git switch -c <新分支名>` 建分支。

## merge 与 rebase：两种整合历史的方式

**merge（合并）**：把两个分支的历史合起来，保留分叉，产生一个合并
提交（两个父提交）。已有 commit 的哈希不变，历史如实记录“这里曾经
分叉、又合起来”。

```bash
git merge feature   # 在 main 上产生一个合并提交
```

**rebase（变基）**：把当前分支的提交一个个重放到目标分支顶端，历史
变成一条直线，但**每个被重放的提交都会生成新的 commit 对象、哈希全
变**。已推送到公共分支的历史不要 rebase：别人的本地仓库还拿着旧哈希，
强推后需要所有人处理分叉。

选择原则：私有分支整理历史用 rebase（保持线性、可读），公共分支整合
用 merge（不改写已发布历史）。

## reflog：HEAD 的移动记录

`reflog` 记录 HEAD 每次移动的位置，找回误删提交的最后一道防线：

```bash
git reflog
# 1ae121d HEAD@{0}: commit: eval: rebuild sample testset
# 3927658 HEAD@{1}: merge: Merge pull request #5
```

误删分支、reset 过头、rebase 改错，都可以从 reflog 里找到旧 commit
哈希再 `git checkout` 或 `git branch` 恢复。reflog 条目默认保留
**90 天**，过期后由 `git gc` 清理；`git reflog expire` 可手动提前清理。

reflog 只存在本地，不随 `push`/`fetch` 同步：别人仓库里的 reflog 看
不到你的操作，丢了的提交只能在自己机器上找回。

## gc：压缩与清理

`git gc` 负责仓库的定期整理：把松散对象（每个对象一个文件）打包成
packfile（增量压缩、去重），清理 unreachable 对象（没有任何引用指向
的旧对象），并修剪过期的 reflog 条目。

`gc` 默认由 Git 在 `commit`、`merge` 等操作后按需自动触发，松散对象
过多时自动打包。手动执行 `git gc --aggressive` 会用更激进的压缩参数
重打全部 pack，耗时长、大仓库上可能跑几分钟，只在仓库明显臃肿时用。
