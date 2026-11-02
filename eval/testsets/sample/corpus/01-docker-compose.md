# Docker Compose 部署笔记

本文记录 RagMate 本地开发环境的 Docker Compose 编排，文件为仓库根目录的
`docker-compose.yml`，用 `docker compose up -d` 启动整套基础设施。

## 服务清单

编排包含四个服务：`web`、`worker`、`db`、`redis`。其中 `db` 与 `redis`
只在开发环境由 Compose 管理，生产环境应改用托管服务。

### web：应用服务

`web` 服务运行 FastAPI 应用，镜像由本地 `Dockerfile` 构建，启动命令为：

```
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

端口映射为 `8000:8000`，即容器内的 8000 端口映射到宿主机的 8000 端口，
浏览器访问 `http://localhost:8000` 即可打开前端页面，接口文档在
`http://localhost:8000/docs`。

重启策略为 `unless-stopped`：容器异常退出时自动重启，但手动 `docker compose
stop` 停止后不会在宿主机重启时自动拉起。`web` 服务通过 `env_file` 读取
`.env` 文件，数据库地址、Redis 地址与 LLM 密钥都在其中配置。

### worker：后台任务

`worker` 服务与 `web` 共用同一个镜像，启动命令不同，负责执行耗时的入库
任务。它的重启策略同样是 `unless-stopped`。`worker` 不对外暴露端口，只通过
Redis 队列与 `web` 协作。

### db：PostgreSQL 15

`db` 服务使用 `postgres:15` 镜像，端口映射为 `5432:5432`，数据保存在具名
卷 `pgdata` 中，删除容器不会丢失数据，只有 `docker volume rm` 会删卷。

健康检查使用 `pg_isready`，配置如下：

```yaml
healthcheck:
  test: ["CMD-SHELL", "pg_isready -U postgres"]
  interval: 10s
  timeout: 5s
  retries: 5
  start_period: 30s
```

即每 10 秒检查一次，单次超时 5 秒，连续失败 5 次判定为不健康，容器启动后
有 30 秒宽限期，宽限期内失败不计入重试。`web` 服务通过
`depends_on` 的 `condition: service_healthy` 等待数据库就绪后再启动，避免
应用在数据库未准备好时反复重连。

重启策略为 `always`：无论何种原因停止，Docker 守护进程都会尝试拉起它。

### redis：缓存与队列

`redis` 服务使用 `redis:7` 镜像，端口映射为 `6379:6379`，启动参数开启
AOF 持久化（`appendonly yes`），重启后可从 AOF 文件恢复数据。`redis` 服务
的重启策略同样是 `always`。

## 网络与卷

四个服务共享一个名为 `backend` 的内部网络，服务之间用服务名作主机名互相
访问，例如 `web` 连接数据库使用的主机名就是 `db`，而不是 `localhost`。
只有 `web` 的 8000 端口、`db` 的 5432 端口和 `redis` 的 6379 端口映射到了
宿主机，`worker` 不对外暴露端口。

具名卷只有一个：`pgdata`，挂载到 `db` 服务的 `/var/lib/postgresql/data`。
Redis 的数据通过 AOF 文件保存在容器内的 `/data` 目录，开发环境不单独挂卷，
重启容器后 AOF 文件随容器层保留，删除容器即丢失。

## 常用命令

```bash
# 启动全部服务（后台运行）
docker compose up -d

# 查看各服务状态，健康检查结果显示在 Status 列
docker compose ps

# 只重启 web 服务
docker compose restart web

# 停止服务但不删除数据卷
docker compose stop

# 停止并删除容器与网络，保留 pgdata 卷
docker compose down
```

`docker compose down -v` 会连同 `pgdata` 卷一起删除，执行前必须确认已经
备份数据库，开发环境误删后只能从备份恢复。

## 环境变量与 secrets

`web` 服务通过 `env_file` 读取 `.env`，Compose 文件里的 `${VAR:-default}`
语法提供默认值：`${DATABASE_URL:-postgres://postgres:postgres@db:5432/ragmate}`
表示环境变量未设置时用冒号后的默认值。`.env` 文件本身不入仓，仓库里只
保留 `.env.example` 作模板，新人复制后填自己的密钥。

敏感值不要写进 `docker-compose.yml`：`LLM_API_KEY` 这类密钥只放 `.env`，
且 `.env` 已在 `.gitignore` 中。需要把密钥挂进容器的场景用 Docker
secrets：密钥以文件形式挂载到 `/run/secrets/`，应用读文件而非读环境变量，
`docker compose config` 输出里也不会出现密钥明文。

`env_file` 与 `environment` 同时指定同一个变量时，`environment` 里的值
优先。调试时用 `docker compose config` 查看合并后的最终配置，确认变量
解析结果符合预期再启动。

## 资源限制与日志

开发环境不设资源限制，生产环境的 `docker-compose.yml` 覆盖文件里给
`web` 服务加了部署约束：

```yaml
deploy:
  resources:
    limits:
      cpus: "2.0"
      memory: 2G
    reservations:
      memory: 512M
```

`limits` 是硬上限，容器超过 2G 内存会被 OOM 杀掉；`reservations` 是调度
预留，宿主机可用内存不足 512M 时容器不会被调度到该节点。`cpus: "2.0"`
表示最多使用两个 CPU 核心的时间配额，不是独占核心。

日志用 json-file 驱动并做轮转，避免单个容器日志文件无限增长：

```yaml
logging:
  driver: json-file
  options:
    max-size: "10m"
    max-file: "3"
```

单个日志文件到 10MB 即轮转，最多保留 3 个历史文件。查日志用
`docker compose logs -f web` 跟随输出，加 `--tail 200` 只看最后 200 行。
`docker compose logs` 默认输出全部服务的日志混在一起，定位问题时先用
服务名过滤。

## 多环境：覆盖文件

开发与生产共用一个 `docker-compose.yml`，差异用覆盖文件表达。默认
情况下，`docker compose` 会自动合并同目录下的 `docker-compose.override.yml`
（如果存在），后者只写与基准不同的字段：

```yaml
# docker-compose.override.yml（开发环境）
services:
  web:
    ports:
      - "8000:8000"
    environment:
      - DEBUG=true
```

生产环境用 `-f` 显式指定多个文件，后指定的文件优先：

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

合并规则是：标量与映射按服务名深度合并，`ports`、`volumes` 这类列表
是替换而非追加——覆盖文件里写了 `ports`，基准文件的端口映射会被整个
替换掉，不是两个都生效。`docker-compose.prod.yml` 里生产环境的 `web`
服务不映射 8000 端口，流量只走反向代理，`depends_on` 与健康检查保持
与基准一致。

覆盖文件只放差异字段，不要把整个服务定义复制一份：复制后基准改了
一处，覆盖文件里的旧定义会悄悄把新配置顶掉，排查时很难发现。
