# CLAUDE.md

本文件为使用 Claude Code (claude.ai/code) 处理本仓库中的代码提供指导。

## 项目概述

这是一个循序渐进的 12 节学习项目，从零开始构建一个类似 Claude Code 的微型智能体。每节课（`s01`–`s12`）都会向核心智能体循环添加一个机制。该仓库包含两部分：Python 参考实现（`agents/`）和一个交互式 Next.js 学习平台（`web/`）。

## 环境设置

复制 `.env` 文件并填写所需值：

```sh

ANTHROPIC_API_KEY=sk-ant-xxx

MODEL_ID=claude-sonnet-4-6

# 可选：覆盖兼容提供商的基本 URL

ANTHROPIC_BASE_URL=https://api.anthropic.com

```

所有 Python 智能体都会从环境变量中读取 `MODEL_ID`，并可选择性地读取 `ANTHROPIC_BASE_URL`。 `.env` 文件支持与 Anthropic 兼容的提供程序（MiniMax、GLM、Kimi、DeepSeek）。

## 命令

### Python 代理

```sh
# 安装依赖项（使用 uv）

pip install -r requirements.txt

# 或使用 uv：

uv sync

# 直接运行任何代理

python agents/s01_agent_loop.py

python agents/s11_autonomous_agents.py

python agents/s12_worktree_task_isolation.py

```

### Web 平台

```sh

cd web

npm install

npm run dev # 在 http://localhost:3000 启动开发服务器（同时运行 extract）

npm run build # 生产环境构建（同时运行 extract）

npm run extract # 重新生成 web/src/data/generated/{versions.json,docs.json}

```

`extract` 脚本（`web/scripts/extract-content.ts`）解析 `agents/*.py` 和 `docs/{en,zh,ja}/*.md`构建到 Next.js 应用使用的预构建 JSON 文件中。

## 架构

### Python 代理 (`agents/`)

每个文件都是一个独立的脚本，包含一个 `agent_loop()` 函数。它们按以下方式逐步构建：

| 会话 | 核心添加 |

|---------|---------------|

| s01 | 最小循环：`while stop_reason == "tool_use"`，包含一个 bash 工具 |

| s02 | 工具分发映射：`name -> handler` |

| s03 | `TodoManager`，包含循环内提醒 |

| s04 | 使用隔离的 `messages[]` 生成的子代理 |

| s05 | `SkillLoader` — 通过 `tool_result` 注入 `SKILL.md` 内容，而不是系统提示 |

| s06 | 三层上下文压缩（微压缩/自动压缩/归档压缩） |

| s07 | 基于文件的 `TaskManager`，带有依赖关系图（状态在 `/compact` 目录下仍然存在） |

| s08 | `BackgroundManager`，带有守护线程和通知队列 |

| s09 | `TeammateManager`，带有基于 JSONL 的异步邮箱 |

| s10 | `request_id` 有限状态机，用于关机和计划审批协议 |

| s11 | 空闲周期 + 自动认领：团队成员轮询共享任务看板 |

| s12 | 工作树生命周期（`WorktreeManager`）+ 任务看板协调 |

所有代理共享相同的 `client = Anthropic(base_url=...)` 初始化模式，并使用 `dotenv` 加载环境变量。

### Web 平台 (`web/`)

Next.js 16 应用，带有 i18n 路由（`[locale]` 段，locales: `en`, `zh`, `ja`）。

**数据流：**

1. `npm run extract` 运行 `web/scripts/extract-content.ts` → 写入 `web/src/data/generated/versions.json` 和 `docs.json`

2. 这些生成的 JSON 文件被提交到代码仓库（在 `agents/` 目录不存在时用作 Vercel 的备用方案）

3. 从生成的 JSON 文件中读取页面；无运行时 API 调用

**`web/src/` 目录下的关键目录：**

- `app/[locale]/(learn)/[version]/` — 每个会话的学习页面，包含源代码查看器、文档、可视化图表和差异比较

- `components/visualizations/s01-s12.tsx` — 每个会话一个动画图表

- `components/simulator/` — 交互式代理循环逐步执行模拟器

- `components/architecture/` — 架构图和执行流程视图

- `lib/constants.ts` — `VERSION_ORDER`、`VERSION_META`、`LAYERS`（会话元数据的单一数据源）

- `types/agent-data.ts` — 共享的 TypeScript 接口（`AgentVersion`、`VersionDiff`、`DocContent` 等）

**国际化：** 服务器端通过 `lib/i18n-server.ts`，客户端通过`lib/i18n.tsx`。区域设置源自 URL 路径段。

### 技能 (`skills/`)

s05 使用的 SKILL.md 文件。每个技能都有一个 YAML 前置元数据 `description` 字段，用于匹配何时加载该技能。`agent-builder` 技能在 `skills/agent-builder/references/` 中包含参考实现。

### 文档 (`docs/`)

三语 (`en/`, `zh/`, `ja/`) Markdown 文件，每个会话一个。格式：问题 → 解决方案 → ASCII 图 → 最小代码。

## 关键约定

- 每个 `agents/s*.py` 文件必须保持独立（不得从其他代理文件导入）。

- 核心的 `agent_loop()` 签名在不同会话中保持不变——仅围绕它添加处理程序和管理器。

- 当 `agents/*.py` 或 `docs/` 文件发生更改时，`web/src/data/generated/` 文件会被提交，必须重新生成（运行 `npm run extract`）。

- `web/src/lib/constants.ts` 中的 `VERSION_META` 是权威注册表；添加新会话需要更新它。

- 没有违反 TypeScript `strict` 模式；该 Web 应用使用 TypeScript 5 和 Next.js 16 应用路由约定。