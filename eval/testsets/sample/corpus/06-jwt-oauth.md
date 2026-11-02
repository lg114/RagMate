# JWT 与 OAuth 2.0 笔记

本文记录 RagMate 后端鉴权的两套机制：自签发的 JWT（API 调用凭证）与
OAuth 2.0 授权码流程（第三方登录）。实现语言为 Python，后端框架 FastAPI。

## JWT 的结构：三段式

JWT 由三部分组成，用点号分隔：

```
header.payload.signature
```

- **header**：声明类型与签名算法，例如 `{"alg": "RS256", "typ": "JWT"}`；
- **payload**：声明（claims），例如 `sub`（用户 id）、`exp`（过期时间戳）、
  `iat`（签发时间）；
- **signature**：对前两段的签名，验签通过才能信任 payload 内容。

三段都用 Base64URL 编码（URL 安全的 Base64，去掉 `=` 填充、把 `+` `/`
换成 `-` `_`），因此 JWT 可以直接放进 `Authorization: Bearer` 头与 URL
查询参数，不需要额外转义。Base64URL 只是编码、不是加密：payload 内容
任何人都能解码看到，不要在里面放密码、密钥等敏感信息。

## HS256 与 RS256：对称与非对称签名

**HS256** 是对称签名：签发与验签用同一个密钥（HMAC-SHA256）。优点是
实现简单、速度快；缺点是任何能验签的服务都能签发，密钥必须只掌握在
签发方手里，多服务共享密钥等于共享签发权。

**RS256** 是非对称签名：私钥签发、公钥验签（RSA-SHA256）。签发方只把
公钥分发给各服务，各服务能验签但不能签发，适合“一个签发方、多个验签
方”的架构。RagMate 用 RS256：认证服务持有私钥，API 服务只存公钥，
公钥泄露不影响安全，私钥泄露才需要轮换。

验签时必须校验 `alg` 字段：攻击者把 `alg` 改成 `none`、去掉 signature
段，验签实现若接受 `none` 算法就会放行无签名 token。验签库要显式
指定允许的算法白名单，拒绝 `none`。

## Access token 短、refresh token 长

Access token 有效期 15 分钟，refresh token 有效期 7 天。分工如下：

- **access token**：每次 API 调用都带，泄露后的影响窗口只有 15 分钟，
  过期自动失效；
- **refresh token**：只在换取新 access token 时用，平时存在 httpOnly
  cookie 里不暴露给 JavaScript，泄露后在服务端吊销该 refresh token，
  攻击者手里的 access token 最多再用 15 分钟。

刷新流程：客户端带 refresh token 请求 `/auth/refresh`，服务端校验
refresh token 有效且未被吊销，签发新的 access token（15 分钟）与新的
refresh token（7 天，旧的作废，即轮换 rotation）。轮换后旧 refresh
token 再被使用，说明可能被盗用：服务端应吊销该用户名下**全部**
refresh token，强制重新登录。

## OAuth 2.0 授权码流程

第三方登录走授权码流程（Authorization Code Flow），步骤如下：

1. 前端把用户重定向到授权服务器的授权端点，带上 `client_id`、
   `redirect_uri`、`scope` 与随机 `state`（防 CSRF）；
2. 用户在授权服务器登录并授权，授权服务器把用户重定向回
   `redirect_uri`，并附上一次性授权码 `code`；
3. **后端**拿 `code` 加 `client_secret` 向授权服务器的 token 端点换取
   access token 与 refresh token；`client_secret` 只存在后端，绝不
   出现在前端代码里；
4. 后端用 access token 拉取用户信息，签发本站的 JWT 会话。

`code` 只能用一次、有效期很短（通常 10 分钟），被拦截后攻击者没有
`client_secret` 也换不到 token。`state` 参数在回调时必须比对一致，
不一致说明回调不是由本站发起的，应拒绝。

## PKCE：给公开客户端的补丁

移动端与纯前端应用没有后端、存不住 `client_secret`，授权码被拦截后
攻击者可直接换 token，PKCE（Proof Key for Code Exchange）堵住这个
缺口：

1. 客户端生成随机字符串 `code_verifier`，算出
   `code_challenge = BASE64URL(SHA256(code_verifier))`（S256 方式）；
2. 授权请求带上 `code_challenge` 与 `code_challenge_method=S256`；
3. 换 token 时带上原始的 `code_verifier`，授权服务器用 S256 重新计算
   并比对 `code_challenge`，一致才发 token。

攻击者只拿到 `code` 与 `code_challenge` 也换不到 token，因为
SHA256 不可逆，推不出 `code_verifier`。RagMate 的桌面端登录用 PKCE
+S256，Web 端用标准授权码流程加 `client_secret`。

## JWT 无法主动失效：靠短有效期与吊销

JWT 一旦签发，在 `exp` 到期前验签都会通过，服务端无法单方面让某个
已签发的 token 立即失效——除非维护黑名单/吊销列表，验签时查库，
那就失去了 JWT 无状态验签的意义。

实际做法是组合拳：access token 有效期压到 15 分钟（泄露窗口小），
refresh token 可吊销（泄露后吊销 refresh，access 最多再活 15 分钟），
高危操作（改密码、改邮箱）触发时吊销该用户全部 refresh token 并要求
重新登录。需要“立即失效单个 access token”的场景，改用服务端会话
（session id 存 Redis，验签时查 Redis），不要硬套 JWT。
