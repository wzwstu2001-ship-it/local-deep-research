# Phase 3: open-webui 接入 LDR + LightRAG 图谱页 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** open-webui 前端接入 LDR（模型后端 + 会话级文档上传）与 LightRAG（全局图谱页直连），并禁用 open-webui 自带 Documents/RAG，完成三项目融合的最后一段。

**Architecture:** 跨两个仓库的改动，按 spec §8.3 四点 + §13.6/§13.7 认证开放问题落地：
- **LDR 仓库**（`local-deep-research`，`feat/lightrag-integration` 分支）：豁免 `openai_compat` 与 `chat_upload` 端点的 Flask session 认证（开放模式，用户已选定），改用固定服务用户名。
- **open-webui 仓库**（`open-webui`，fork，`main` 为 base，需开新 feature 分支）：① 连接 LDR 作为 OpenAI 兼容模型后端并透传 `chat_id`；② 新增会话级上传组件（透传 `chat_id`）；③ 新增图谱页直连 LightRAG；④ 禁用自带 Documents/RAG；⑤ 端口从 8080 改开。

**Tech Stack:** Svelte 5 + TypeScript + Tailwind（open-webui 前端）、Python FastAPI（open-webui 后端）、Flask（LDR）、FastAPI（LightRAG）。

**Spec:** `docs/superpowers/specs/2026-08-28-ldr-lightrag-openwebui-integration-design.md`（本计划从其 §7/§8.3/§12/§13 论证）

## Global Constraints

- **认证开放模式（用户已选定）**：LDR `openai_compat` 与 `chat_upload` 端点豁免 Flask session 认证，改用固定服务用户名 `LDR_OPENAI_COMPAT_USERNAME`（默认 `"openwebui"`）；安全边界依赖 LDR 服务仅绑定 `127.0.0.1`。不新增 API key 校验、不引入服务账号。
- **图谱开放模式（用户已选定）**：LightRAG 图 API 前端直连（`combined_auth` 在未启用认证时放行），不经 open-webui 后端转发、不经 LDR 代理。
- **fail-closed（spec §9）**：`chat_id` 缺失/不可解析 → 明确 400/404 或空结果，**绝不回退全库检索**。
- **上传端点**：用 Phase 2 已落地的 `POST /library/api/collections/chat/upload`（form `chat_id` + `files`），**不是** spec §7.1 字面写的 `/api/v1` upload（见 Ruling 1）。
- **提交信息英文**，结尾 `Co-Authored-By: Claude Code <noreply@anthropic.com>`。
- **仓库归属**：Task 1 在 LDR 仓库；Task 2–6 在 open-webui 仓库。两个仓库各自开分支、各自提交。

---

## Rulings（写计划时裁定，spec 未决处）

- **Ruling 1（上传端点）**：spec §7.1/§8.3 写「LDR `/api/v1` upload」，但 Phase 2 实际落地的是 `/library/api/collections/chat/upload`（`POST`，form `chat_id` + `files`，`@login_required` → 本计划 Task 1 豁免）。Phase 3 上传组件统一调此端点。成本：若将来要统一为 `/api/v1` 命名，需同步改 Phase 2 的 Task 7 端点与测试。
- **Ruling 2（chat_id 透传方式）**：open-webui 后端 `openai.py::generate_chat_completion` 第 1492 行把 `metadata` pop 出 payload，导致 LDR 读不到 `metadata.chat_id`。方案：pop 后，若 `metadata` 含 `chat_id`，写回 `payload['chat_id']`，**仅当连接 URL 非 `api.openai.com`**（复用第 1551 行既有判断）。理由：LDR 读 body 顶层 `chat_id`；本地 OpenAI 兼容后端（Ollama/vLLM/LDR）通常忽略未知字段；官方 OpenAI 严格校验故排除。成本：若某本地后端拒绝 `chat_id` 字段，需收敛为 `api_config` 标记识别 LDR。
- **Ruling 3（上传组件落地）**：新增独立上传入口组件，而非改造 `MessageInput.svelte` 的复杂上传状态机。在聊天页头部/输入区上方加「上传到会话」按钮，fetch 直连 LDR，`FormData` 带 `chat_id: $chatId` + `files`，展示 toast。侵入最小。
- **Ruling 4（图谱页 MVP）**：图谱页不引入新图渲染依赖，用「标签列表 + 选择标签后展示实体/关系列表」的轻量实现（纯 Svelte + Tailwind）。完整力导向图作为后续增强。

---

### Task 1: LDR 开放模式认证豁免（openai_compat + chat_upload）

> 仓库：**LDR**（`local-deep-research`，分支 `feat/lightrag-integration`）

**Files:**
- Modify: `src/local_deep_research/web/api.py`（新增 `get_openai_compat_username()`）
- Modify: `src/local_deep_research/web/routes/openai_compat_routes.py`（去 `@api_access_control`，用固定服务用户名）
- Modify: `src/local_deep_research/research_library/routes/rag_routes.py:2120-2146`（去 `@login_required`，用固定服务用户名）
- Test: `tests/web/test_openai_compat_routes.py`、`tests/research_library/routes/test_rag_routes_chat_upload.py`

**Interfaces:**
- Consumes: 现有 `get_current_username()`（`security/rate_limiter.py:188`，从 `g.current_user`/session 取）；`api_access_control`（`web/api.py:188`，无 username 返回 401）；`login_required`（chat_upload 装饰器）。
- Produces: `get_openai_compat_username() -> str`（读 env `LDR_OPENAI_COMPAT_USERNAME`，默认 `"openwebui"`）。两个端点此后可在无 session cookie 的 server-to-server 调用下运行，`quick_summary` 走默认设置（`_load_user_context_into_params` 在 username 存在但无 DB 时返回空 snapshot，或直接用空设置）。

- [ ] **Step 1: 写失败测试**

在 `tests/web/test_openai_compat_routes.py` 追加「无认证 session 也放行」的测试（当前 `client` fixture 若依赖 session 认证，需改为裸 `app.test_client()`）：

```python
def test_chat_completions_open_mode_without_session(app):
    """Open mode: no Flask session cookie — a headless call must not 401."""
    client = app.test_client()
    with patch(
        "local_deep_research.web.routes.openai_compat_routes._load_user_context_into_params",
        side_effect=lambda p, u, **kw: p.update(username=u, settings_snapshot={}) or None,
    ), patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_svc, patch(
        "local_deep_research.api.research_functions.quick_summary",
        return_value={"summary": "ok"},
    ):
        mock_svc.return_value.get_or_create_session_collection.return_value = "col-abc"
        resp = client.post(
            "/v1/chat/completions",
            json={"model": "ldr", "chat_id": "chat-1",
                  "messages": [{"role": "user", "content": "hi"}]},
        )
    assert resp.status_code == 200
```

在 `tests/research_library/routes/test_rag_routes_chat_upload.py` 追加「无认证 session 也放行」（当前用 `_auth_client`，改为裸 client + patch 服务）：

```python
def test_chat_upload_open_mode_without_session(app):
    """Open mode: chat_upload must run without a login session."""
    client = app.test_client()
    with patch(
        "local_deep_research.chat.service.ChatService"
    ) as mock_svc, patch(
        "local_deep_research.research_library.routes.rag_routes.upload_to_collection"
    ) as mock_upload:
        mock_svc.return_value.get_or_create_session_collection.return_value = "col-abc"
        mock_upload.return_value = ({"success": True}, 200)
        resp = client.post(
            "/library/api/collections/chat/upload",
            data={"chat_id": "chat-1", "files": [(BytesIO(b"x"), "a.txt")]},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    mock_svc.return_value.get_or_create_session_collection.assert_called_once_with("chat-1")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `pytest tests/web/test_openai_compat_routes.py tests/research_library/routes/test_rag_routes_chat_upload.py -v`
Expected: 新增测试 FAIL（当前 `api_access_control` 返回 401 / `login_required` 重定向）。

- [ ] **Step 3: 实现 `get_openai_compat_username()`**

在 [web/api.py](src/local_deep_research/web/api.py) 顶部（`import` 区后，`_scrub_error_fields` 前）新增：

```python
import os


def get_openai_compat_username() -> str:
    """Service username for headless open-webui calls (open mode).

    open-webui is server-to-server and has no Flask session. In open mode
    these endpoints run under a fixed service user whose settings are empty
    (permissive defaults) unless the operator provisions that user. The
    safety boundary is that the LDR server only binds 127.0.0.1.
    """
    return os.environ.get("LDR_OPENAI_COMPAT_USERNAME", "openwebui")
```

- [ ] **Step 4: 改 openai_compat 端点**

在 [openai_compat_routes.py](src/local_deep_research/web/routes/openai_compat_routes.py)：

(1) import 增加 `get_openai_compat_username`：

```python
from ..api import (
    _load_user_context_into_params,
    _scrub_error_fields,
    api_access_control,
    get_openai_compat_username,
)
```

(2) 移除 `@api_access_control` 装饰器（保留 `@api_rate_limit` 与 `@require_json_body`）：

```python
@openai_compat_bp.route("/chat/completions", methods=["POST"])
@api_rate_limit
@require_json_body(error_message="messages are required")
def chat_completions():
```

(3) 把函数体内的 `username = get_current_username()` 改为：

```python
        username = get_openai_compat_username()
```

（`get_current_username` 的 import 若不再使用，从 `from ...security.rate_limiter import ...` 中移除，避免 lint 告警。）

- [ ] **Step 5: 改 chat_upload 端点**

在 [rag_routes.py:2120-2146](src/local_deep_research/research_library/routes/rag_routes.py#L2120-L2146)：

(1) 移除 `@login_required` 装饰器（保留 `@upload_rate_limit_user` / `@upload_rate_limit_ip`）：

```python
@rag_bp.route("/api/collections/chat/upload", methods=["POST"])
@upload_rate_limit_user
@upload_rate_limit_ip
def chat_upload():
```

(2) 把 `ChatService(session["username"])` 改为：

```python
    from ...chat.service import ChatService, ChatSessionNotFound
    from ...web.api import get_openai_compat_username

    try:
        collection_id = ChatService(
            get_openai_compat_username()
        ).get_or_create_session_collection(chat_id)
```

> 注意：`@upload_rate_limit_user` 内部可能依赖 `session["username"]`，若开放模式下抛 KeyError，实现者需确认其 fallback（通常对缺失 username 用 IP 限流）。若会抛错，同样替换为 `get_openai_compat_username()`，并在 Step 4 的测试里覆盖一次上传路径。

- [ ] **Step 6: 运行测试确认通过**

Run: `pytest tests/web/test_openai_compat_routes.py tests/research_library/routes/test_rag_routes_chat_upload.py -v`
Expected: 全部 PASS（含既有用例与新增开放模式用例）。

- [ ] **Step 7: Commit**

```bash
git add src/local_deep_research/web/api.py \
        src/local_deep_research/web/routes/openai_compat_routes.py \
        src/local_deep_research/research_library/routes/rag_routes.py \
        tests/web/test_openai_compat_routes.py \
        tests/research_library/routes/test_rag_routes_chat_upload.py
git commit -m "feat(openai-compat): open-mode auth exemption for server-to-server

open-webui calls /v1/chat/completions and the chat upload endpoint without a
Flask session cookie. Drop api_access_control / login_required from both and
run under a fixed service username (LDR_OPENAI_COMPAT_USERNAME, default
'openwebui') so headless calls work. Safety boundary is 127.0.0.1 binding.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 2: open-webui 后端连接 LDR + 透传 chat_id

> 仓库：**open-webui**（fork，先 `git checkout -b feat/lightrag-integration`）

**Files:**
- Modify: `backend/open_webui/routers/openai.py:1491-1567`（pop metadata 后写回 `chat_id`）
- Modify: `backend/open_webui/env.py`（若需在 env 注释中说明；否则跳过）
- Doc: `.env.example` 增补 `OPENAI_API_BASE_URLS` / `OPENAI_API_KEYS` 说明

**Interfaces:**
- Consumes: `generate_chat_completion` 第 1491-1492 行 `payload = {**form_data}` + `metadata = payload.pop('metadata', None)`；第 1551 行 `elif 'api.openai.com' not in url:` 判断。
- Produces: 对非官方 OpenAI 连接，`payload['chat_id'] = metadata['chat_id']`（若存在），使 LDR `/v1/chat/completions` 能读到 `data.get("chat_id")`。

- [ ] **Step 1: 写失败测试**

open-webui 后端无成熟 pytest 约定，本任务以 **编译 + lint** 作为门（RED/GREEN 由手动 curl 到 LDR 验证 `chat_id` 是否透传）。RED 表现为改动前 LDR 收不到 `chat_id`。

- [ ] **Step 2: 记录基线**

Run: `cd backend && python -m compileall open_webui/routers/openai.py`
Expected: 编译通过（语法门）。

- [ ] **Step 3: 实现 chat_id 写回**

在第 1534 行 `url, key, api_config = await get_openai_connection(idx)` 之后、第 1560 行之前插入（`metadata` 已在第 1492 行 pop，`url` 此时已解析）：

```python
    # Phase 3 (LDR integration): restore the conversation id into the payload
    # body for non-OpenAI backends. LDR's OpenAI-compat endpoint reads
    # `chat_id` from the body to scope retrieval to the chat's collection.
    if metadata and metadata.get('chat_id') and 'api.openai.com' not in url:
        payload['chat_id'] = metadata['chat_id']
```

- [ ] **Step 4: 构建 + lint 验证**

Run: `cd backend && python -m compileall open_webui/routers/openai.py && cd .. && bun run lint`
Expected: 编译通过、lint 通过。

- [ ] **Step 5: 在 .env.example 增补连接说明**

在 open-webui 的 `.env.example` 的 OpenAI 连接段增补：

```
# Phase 3 — connect open-webui to LDR's OpenAI-compatible endpoint.
# LDR serves /v1/chat/completions on http://127.0.0.1:5000/v1 (open mode).
OPENAI_API_BASE_URLS="http://127.0.0.1:5000/v1"
# Key is unused in open mode; a placeholder satisfies the key list.
OPENAI_API_KEYS=""
```

- [ ] **Step 6: Commit**

```bash
git add backend/open_webui/routers/openai.py .env.example
git commit -m "feat(openai): forward chat_id to non-OpenAI backends for LDR

LDR's OpenAI-compat endpoint scopes retrieval to a chat's collection by
reading chat_id from the body. open-webui carries chat_id in metadata and
pops it before proxying; restore it as a top-level field for non-OpenAI
backends (local backends ignore unknown fields).

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 3: open-webui 上传组件（透传 chat_id 到 LDR）

> 仓库：**open-webui**

**Files:**
- Create: `src/lib/components/chat/LdrUpload.svelte`（新上传入口组件）
- Modify: `src/lib/components/chat/MessageInput.svelte`（在输入区上方挂载 `<LdrUpload />`）
- Modify: `src/lib/constants.ts`（增补 `LDR_BASE_URL`）
- Test: `src/lib/components/chat/LdrUpload.test.ts`（bun test，测试导出纯函数）

**Interfaces:**
- Consumes: `chatId` store（`src/lib/stores/index.ts:54`，`$chatId`）；`toast` 工具；`LDR_BASE_URL`（常量，默认 `http://127.0.0.1:5000`）。
- Produces: `POST {LDR_BASE_URL}/library/api/collections/chat/upload`，`FormData` 带 `chat_id` + `files[]`，成功 toast、失败 toast（含 400「chat_id required」/ 404「chat not found」）。

- [ ] **Step 1: 写失败测试**

新建 `src/lib/components/chat/LdrUpload.test.ts`，测试导出的纯函数 `buildLdrUploadForm`（避免 Svelte 渲染栈依赖）：

```ts
import { describe, it, expect } from 'bun:test';
import { buildLdrUploadForm } from './LdrUpload.svelte';

describe('buildLdrUploadForm', () => {
  it('appends chat_id and every file to the FormData', () => {
    const files = [new File(['a'], 'a.txt'), new File(['b'], 'b.txt')];
    const fd = buildLdrUploadForm('chat-1', files);
    expect(fd.get('chat_id')).toBe('chat-1');
    expect(fd.getAll('files')).toHaveLength(2);
  });
});
```

- [ ] **Step 2: 运行测试确认失败**

Run: `bun test src/lib/components/chat/LdrUpload.test.ts`
Expected: FAIL（`buildLdrUploadForm` 未导出）。

- [ ] **Step 3: 实现上传组件**

新建 [LdrUpload.svelte](src/lib/components/chat/LdrUpload.svelte)：

```svelte
<script lang="ts">
  import { chatId } from '$lib/stores';
  import { toast } from 'svelte-sonner';
  import { LDR_BASE_URL } from '$lib/constants';

  let uploading = false;

  export function buildLdrUploadForm(chatId: string, files: File[]): FormData {
    const fd = new FormData();
    fd.append('chat_id', chatId);
    for (const f of files) fd.append('files', f);
    return fd;
  }

  async function onPick(e: Event) {
    const input = e.target as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    input.value = '';
    if (!files.length) return;
    if (!$chatId) {
      toast.error('Start or select a chat before uploading documents.');
      return;
    }
    uploading = true;
    try {
      const fd = buildLdrUploadForm($chatId, files);
      const res = await fetch(`${LDR_BASE_URL}/library/api/collections/chat/upload`, {
        method: 'POST',
        body: fd
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        toast.error(body?.error ?? `Upload failed (${res.status})`);
        return;
      }
      toast.success(`Uploaded ${files.length} document(s) to this chat.`);
    } catch (e) {
      toast.error(`Upload error: ${e}`);
    } finally {
      uploading = false;
    }
  }
</script>

<label class="ldr-upload">
  <input type="file" multiple on:change={onPick} class="hidden" />
  <button type="button" disabled={uploading} on:click={() => (document.querySelector('input[type=file]') as HTMLInputElement)?.click()}>
    {uploading ? 'Uploading…' : 'Upload to chat'}
  </button>
</label>
```

> 实际 button/input 交互写法以 fork 现有组件风格为准（参考 `MessageInput.svelte` 的附件按钮）。

在 [constants.ts](src/lib/constants.ts) 增补：

```ts
export const LDR_BASE_URL = (import.meta.env.VITE_LDR_BASE_URL as string) ?? 'http://127.0.0.1:5000';
```

在 [MessageInput.svelte](src/lib/components/chat/MessageInput.svelte) 输入区上方挂载 `<LdrUpload />`（找到输入框容器，置于其上方一行，具体锚点由实现者按组件结构定位）。

- [ ] **Step 4: 测试 + 构建通过**

Run: `bun test src/lib/components/chat/LdrUpload.test.ts && bun run build`
Expected: 测试 PASS、构建通过。

- [ ] **Step 5: Commit**

```bash
git add src/lib/components/chat/LdrUpload.svelte \
        src/lib/components/chat/LdrUpload.test.ts \
        src/lib/components/chat/MessageInput.svelte \
        src/lib/constants.ts
git commit -m "feat(chat): add chat-scoped upload to LDR

Add an upload entry that posts files to LDR's chat upload endpoint with the
current chat_id, so user documents land in the chat's dedicated collection.
Missing chat_id is surfaced as an error (fail-closed).

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 4: open-webui 图谱页（直连 LightRAG）+ 侧边栏入口

> 仓库：**open-webui**

**Files:**
- Create: `src/routes/(app)/graph/+page.svelte`（图谱页）
- Modify: `src/lib/components/layout/Sidebar.svelte`（`DEFAULT_PINNED_ITEMS` / `isMenuItemVisible` / `getMenuItemMeta` / `menuItemPathPrefixes` 加 `graph`）
- Modify: `src/lib/components/layout/Sidebar/UserMenu.svelte`（加图谱链接块）
- Modify: `src/lib/constants.ts`（增补 `LIGHTRAG_BASE_URL`）

**Interfaces:**
- Consumes: LightRAG `GET /graph/label/list`（graph_routes.py:170，返回标签列表）、`GET /graphs?label=&max_depth=&max_nodes=`（graph_routes.py:231，返回子图）。开放模式前端直连，无需认证头。
- Produces: 图谱页：加载标签列表 → 选择标签 → 拉取子图 → 展示实体/关系（列表形式，Ruling 4 MVP）。

- [ ] **Step 1: 写失败测试**

以 `bun run build` 的 TS 编译失败为 RED（路由页不存在会导致构建或导航报错；Svelte 路由页通常无独立单测）。RED 表现为：新增路由前 `/graph` 无页面。

- [ ] **Step 2: 记录基线**

Run: `bun run build`
Expected: 通过（基线）。

- [ ] **Step 3: 增补常量 + 图谱页**

在 [constants.ts](src/lib/constants.ts) 增补：

```ts
export const LIGHTRAG_BASE_URL = (import.meta.env.VITE_LIGHTRAG_BASE_URL as string) ?? 'http://127.0.0.1:9621';
```

新建 [src/routes/(app)/graph/+page.svelte](src/routes/(app)/graph/+page.svelte)：

```svelte
<script lang="ts">
  import { onMount } from 'svelte';
  import { LIGHTRAG_BASE_URL } from '$lib/constants';

  let labels: string[] = [];
  let selected = '';
  let graph: { nodes?: any[]; edges?: any[] } | null = null;
  let loading = false;
  let error = '';

  async function loadLabels() {
    try {
      const res = await fetch(`${LIGHTRAG_BASE_URL}/graph/label/list`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      labels = data?.labels ?? data ?? [];
    } catch (e) {
      error = `Failed to load labels: ${e}`;
    }
  }

  async function loadGraph() {
    if (!selected) return;
    loading = true;
    error = '';
    try {
      const res = await fetch(
        `${LIGHTRAG_BASE_URL}/graphs?label=${encodeURIComponent(selected)}&max_depth=1&max_nodes=100`
      );
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      graph = await res.json();
    } catch (e) {
      error = `Failed to load graph: ${e}`;
    } finally {
      loading = false;
    }
  }

  onMount(loadLabels);
</script>

<div class="graph-page p-4">
  <h1 class="text-lg font-semibold mb-4">Knowledge Graph</h1>
  <div class="flex gap-2 mb-4">
    <select bind:value={selected} on:change={loadGraph}>
      <option value="">Select a label…</option>
      {#each labels as label}
        <option value={label}>{label}</option>
      {/each}
    </select>
  </div>
  {#if error}<p class="text-red-500">{error}</p>{/if}
  {#if loading}<p>Loading…</p>{/if}
  {#if graph}
    <div class="grid grid-cols-2 gap-4">
      <section>
        <h2 class="font-medium">Entities ({graph.nodes?.length ?? 0})</h2>
        <ul>
          {#each graph.nodes ?? [] as node}
            <li class="border rounded p-2 mb-1">{node}</li>
          {/each}
        </ul>
      </section>
      <section>
        <h2 class="font-medium">Relations ({graph.edges?.length ?? 0})</h2>
        <ul>
          {#each graph.edges ?? [] as edge}
            <li class="border rounded p-2 mb-1">{edge}</li>
          {/each}
        </ul>
      </section>
    </div>
  {/if}
</div>
```

> 节点/边字段名以 LightRAG `/graphs` 实际返回为准，实现者需在联调时用 curl 确认并适配（`nodes`/`edges` 的字段可能是 `entities`/`relationships`）。

- [ ] **Step 4: 侧边栏加入口**

在 [Sidebar.svelte](src/lib/components/layout/Sidebar.svelte)：

(1) `DEFAULT_PINNED_ITEMS` 加 `'graph'`（第 97 行）：

```ts
const DEFAULT_PINNED_ITEMS = ['notes', 'workspace', 'graph'];
```

(2) `isMenuItemVisible` 加 case（第 157-188 行 switch 内）：

```ts
case 'graph':
  return $user?.role === 'admin';
```

(3) `getMenuItemMeta` 加项（第 190-199 行 items 内）：

```ts
graph: { label: 'Graph', href: '/graph', iconType: 'workspace' },
```

(4) `menuItemPathPrefixes` 加项（第 201-207 行）：

```ts
graph: '/graph',
```

在 [UserMenu.svelte](src/lib/components/layout/Sidebar/UserMenu.svelte) 的 Workspace 链接块附近（第 241-283 行）加一个「Knowledge Graph」链接块，`href="/graph"`（复用现有链接块的 class 结构）。

- [ ] **Step 5: 构建验证**

Run: `bun run build && bun run lint`
Expected: 构建通过、lint 通过。

- [ ] **Step 6: Commit**

```bash
git add "src/routes/(app)/graph/+page.svelte" \
        src/lib/components/layout/Sidebar.svelte \
        src/lib/components/layout/Sidebar/UserMenu.svelte \
        src/lib/constants.ts
git commit -m "feat(graph): add LightRAG graph page with sidebar entry

Add a /graph route that queries LightRAG's graph API directly (label list +
subgraph) in open mode, plus a sidebar/UserMenu navigation entry.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 5: 禁用 open-webui 自带 Documents/RAG

> 仓库：**open-webui**

**Files:**
- Modify: `src/lib/components/layout/Sidebar.svelte`（`isMenuItemVisible` 让 `workspace` 的知识库子项不可见）
- Modify: `src/lib/components/layout/Sidebar/UserMenu.svelte`（移除/隐藏 Knowledge 链接块）

**Interfaces:**
- Consumes: `isMenuItemVisible('workspace')`（Sidebar.svelte:164-172）；UserMenu 的 Knowledge 链接块。
- Produces: 前端不再展示 Documents/Knowledge 入口，用户上传改走 Task 3 的 LDR 上传。

- [ ] **Step 1: 记录基线**

Run: `bun run build`
Expected: 通过。

- [ ] **Step 2: 隐藏入口**

在 [Sidebar.svelte](src/lib/components/layout/Sidebar.svelte) 的 `isMenuItemVisible`，把 `workspace` case 改为隐藏（取决于 fork 是否还要保留其它 workspace 子项）：

```ts
case 'workspace':
  // Phase 3: self-hosted RAG is replaced by LDR + LightRAG. Hide the
  // built-in Documents/RAG entry.
  return false;
```

在 [UserMenu.svelte](src/lib/components/layout/Sidebar/UserMenu.svelte)，移除或注释掉指向 `/workspace/knowledge`（或类似）的链接块。

> 若 fork 把「Workspace」拆成多个子项（Models/Knowledge/Prompts/Tools），则只隐藏 Knowledge 子项，保留其余。实现者需按实际 UserMenu 结构精确处理，避免误删 Models/Prompts 入口。

- [ ] **Step 3: 构建验证**

Run: `bun run build && bun run lint`
Expected: 构建通过、lint 通过。

- [ ] **Step 4: Commit**

```bash
git add src/lib/components/layout/Sidebar.svelte \
        src/lib/components/layout/Sidebar/UserMenu.svelte
git commit -m "feat(workspace): hide built-in Documents/RAG entry

Uploads now go to LDR (chat-scoped) and global docs to LightRAG; hide the
built-in Documents/RAG entry to avoid a second, unintegrated upload path.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

### Task 6: open-webui 端口配置（8080 冲突 → 非 8080）

> 仓库：**open-webui**

**Files:**
- Modify: `backend/start.sh:34`（若需改默认端口；否则仅文档）
- Doc: `.env.example` 增补端口说明

**Interfaces:**
- Consumes: `backend/start.sh:34` `PORT="${PORT:-8080}"`。
- Produces: 部署时 open-webui 改用非 8080（如 8081），避免与 llama.cpp 冲突。

- [ ] **Step 1: 改默认端口 + 文档**

在 [backend/start.sh:34](backend/start.sh#L34) 把默认端口改为 8081：

```bash
PORT="${PORT:-8081}"
```

在 `.env.example` 增补：

```
# Phase 3 — open-webui default 8080 collides with llama.cpp; use 8081.
PORT=8081
```

- [ ] **Step 2: 验证**

Run: `bash -n backend/start.sh`
Expected: 语法通过。

- [ ] **Step 3: Commit**

```bash
git add backend/start.sh .env.example
git commit -m "chore(server): default port 8081 to avoid llama.cpp collision

open-webui's default 8080 conflicts with llama.cpp; default to 8081.

Co-Authored-By: Claude Code <noreply@anthropic.com>"
```

---

## 测试运行汇总（执行者快速参照）

只跑 mirror 变更模块的目录，不跑全量：

```bash
# Task 1（LDR 仓库）
pytest tests/web/test_openai_compat_routes.py tests/research_library/routes/test_rag_routes_chat_upload.py -v

# Task 2-6（open-webui 仓库）
cd backend && python -m compileall open_webui/routers/openai.py && cd ..
bun run lint
bun run build
bun test src/lib/components/chat/LdrUpload.test.ts   # Task 3
```

跨切面回归（里程碑时跑）：LDR 侧 `pytest tests/web tests/research_library/routes -v`；open-webui 侧 `bun run build && bun run lint`。
