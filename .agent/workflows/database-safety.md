---
description: 数据库安全操作规则 - 每次涉及数据库或 git 操作前必须检查
---

# 🚨 数据库安全规则 - 必须严格遵守

## 绝对禁止的操作

1. **🚨 绝对禁止 `git reset --hard` — 这会用远端小 db 覆盖本地大 db，丢失全部 backfill 数据**
   - 本地 db（4GB+）包含 2000 万行 backfill 数据，远端 db 只有几万行
   - 任何 `git reset`、`git checkout -- db/`、`git restore` 都会导致数据丢失

2. **禁止在 backfill/写入进程运行时对 db 文件执行任何 git 操作**
   - 不能 `git checkout` db 文件
   - 不能 `git stash` 包含 db 文件的改动
   - 不能 `git reset` db 文件
   - 不能 `git pull --rebase` 当本地有 db 文件变更时

3. **禁止在没有备份的情况下修改 db 文件**
   - 任何可能影响 db 的操作前，先 `cp db/xxx.db db/xxx.db.bak_$(date +%Y%m%d_%H%M%S)`

4. **禁止直接删除或覆盖任何 .db 文件**

5. **推代码时只 git add 特定 .py/.yml/.md 文件，绝不用 git add -A 或 git add .**

## 操作前检查清单

每次涉及 git 或数据库操作前，必须执行：

```bash
# 1. 检查是否有进程在写 db
ps aux | grep -E "backfill|scan_service|pipeline" | grep -v grep

# 2. 检查 WAL 文件是否存在（说明有活跃写入）
ls -la db/*.db-wal 2>/dev/null

# 3. 如果以上有输出，绝对不要动 db 文件
```

## Git 操作安全流程

当需要 push 代码改动（不涉及 db）时：

```bash
# 1. 只 add 需要的文件，不要 git add -A
git add versions/v2/scripts/xxx.py versions/v3/scripts/xxx.py

# 2. commit
git commit -m "message"

# 3. 如果远端有冲突，用 force push（仅限 py 文件改动）
git push --force-with-lease
```

## 恢复损坏数据库

```bash
# 使用 .recover 命令
sqlite3 db/damaged.db ".recover" | sqlite3 db/recovered.db

# 验证
sqlite3 db/recovered.db "SELECT COUNT(*) FROM scan_results;"

# 替换
mv db/damaged.db db/damaged.db.corrupt
mv db/recovered.db db/damaged.db
```

## 核心原则

> **数据是最宝贵的资产。任何操作的便利性都不值得拿数据安全冒险。**
> **如果不确定操作是否安全，就不要做。先问用户。**
