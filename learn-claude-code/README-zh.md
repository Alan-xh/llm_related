# Learn Claude Code -- 从 0 到 1 构建 nano Claude Code-like agent

[English](./README.md) | [中文](./README-zh.md) | [日本語](./README-ja.md)

```
                    THE AGENT PATTERN
                    =================

    User --> messages[] --> LLM --> response
                                      |
                            stop_reason == "tool_use"?
                           /                          \
                         yes                           no
                          |                             |
                    execute tools                    return text
                    append results
                    loop back -----------------> messages[]


    这是最小循环。每个 AI 编程 Agent 都需要这个循环。
    生产级 Agent 还会叠加策略、权限与生命周期层。
```

**12 个递进式课程, 从简单循环到隔离化的自治执行。**
**每个课程添加一个机制。每个机制有一句格言。**

> **s01** &nbsp; *"Bash 就够了"* &mdash; 一个工具 + 一个循环 = 一个智能体
>
> **s02** &nbsp; *"循环没有变"* &mdash; 加工具就是加 handler, 不是重写循环
>
> **s03** &nbsp; *"先计划再行动"* &mdash; 可见的计划提升任务完成率
>
> **s04** &nbsp; *"进程隔离 = 上下文隔离"* &mdash; 每个子智能体独立 messages[]
>
> **s05** &nbsp; *"按需加载, 而非预装"* &mdash; 通过 tool_result 注入知识, 而非塞进 system prompt
>
> **s06** &nbsp; *"策略性遗忘"* &mdash; 忘掉旧上下文, 换来无限会话
>
> **s07** &nbsp; *"状态在压缩后存活"* &mdash; 文件持久化的状态不怕上下文压缩
>
> **s08** &nbsp; *"发射后不管"* &mdash; 非阻塞线程 + 通知队列
>
> **s09** &nbsp; *"追加即发送, 排空即读取"* &mdash; 异步邮箱实现持久化队友通信
>
> **s10** &nbsp; *"同一个 request_id, 两个协议"* &mdash; 一个 FSM 模式驱动关机 + 计划审批
>
> **s11** &nbsp; *"轮询, 认领, 工作, 重复"* &mdash; 无需协调者, 智能体自组织
>
> **s12** &nbsp; *"目录隔离, 任务 ID 协调"* &mdash; 任务板协调 + 按需 worktree 隔离通道

---

## 核心模式

```python
def agent_loop(messages):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM,
            messages=messages, tools=TOOLS,
        )
        messages.append({"role": "assistant",
                         "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = TOOL_HANDLERS[block.name](**block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

每个课程在这个循环之上叠加一个机制 -- 循环本身始终不变。

## 范围说明 (重要)

本仓库是一个 0->1 的学习型项目，用于从零构建 nano Claude Code-like agent。
为保证学习路径清晰，仓库有意简化或省略了部分生产机制：

- 完整事件 / Hook 总线 (例如 PreToolUse、SessionStart/End、ConfigChange)。  
  s12 仅提供教学用途的最小 append-only 生命周期事件流。
- 基于规则的权限治理与信任流程
- 会话生命周期控制 (resume/fork) 与更完整的 worktree 生命周期控制
- 完整 MCP 运行时细节 (transport/OAuth/资源订阅/轮询)

仓库中的团队 JSONL 邮箱协议是教学实现，不是对任何特定生产内部实现的声明。

## 快速开始

```sh
git clone https://github.com/shareAI-lab/learn-claude-code
cd learn-claude-code
pip install -r requirements.txt
cp .env.example .env   # 编辑 .env 填入你的 ANTHROPIC_API_KEY

python agents/s01_agent_loop.py       # 从这里开始
python agents/s11_autonomous_agents.py  # 完整自治团队
python agents/s12_worktree_task_isolation.py  # Task 感知的 worktree 隔离
```

### Web 平台

交互式可视化、分步动画、源码查看器, 以及每个课程的文档。

```sh
cd web && npm install && npm run dev   # http://localhost:3000
```

## 学习路径

```
第一阶段: 循环                       第二阶段: 规划与知识
==================                   ==============================
s01  Agent 循环              [1]     s03  TodoWrite               [5]
     while + stop_reason                  TodoManager + nag 提醒
     |                                    |
     +-> s02  工具               [4]     s04  子智能体             [5]
              dispatch map: name->handler     每个子智能体独立 messages[]
                                              |
                                         s05  Skills               [5]
                                              SKILL.md 通过 tool_result 注入
                                              |
                                         s06  Compact              [5]
                                              三层上下文压缩

第三阶段: 持久化                     第四阶段: 团队
==================                   =====================
s07  任务系统                [8]     s09  智能体团队             [9]
     文件持久化 CRUD + 依赖图             队友 + JSONL 邮箱
     |                                    |
s08  后台任务                [6]     s10  团队协议               [12]
     守护线程 + 通知队列                  关机 + 计划审批 FSM
                                          |
                                     s11  自治智能体             [14]
                                          空闲轮询 + 自动认领
                                     |
                                     s12  Worktree 隔离          [16]
                                          任务协调 + 按需隔离执行通道

                                     [N] = 工具数量
```

## 项目结构

```
learn-claude-code/
|
|-- agents/                        # Python 参考实现 (s01-s12 + 完整版)
|-- docs/{en,zh,ja}/               # 心智模型优先的文档 (3 种语言)
|-- web/                           # 交互式学习平台 (Next.js)
|-- skills/                        # s05 的 Skill 文件
+-- .github/workflows/ci.yml      # CI: 类型检查 + 构建
```

## 文档

心智模型优先: 问题、方案、ASCII 图、最小化代码。
[English](./docs/en/) | [中文](./docs/zh/) | [日本語](./docs/ja/)

| 课程 | 主题 | 格言 |
|------|------|------|
| [s01](./docs/zh/s01-the-agent-loop.md) | Agent 循环 | *Bash 就够了* |
| [s02](./docs/zh/s02-tool-use.md) | 工具 | *循环没有变* |
| [s03](./docs/zh/s03-todo-write.md) | TodoWrite | *先计划再行动* |
| [s04](./docs/zh/s04-subagent.md) | 子智能体 | *进程隔离 = 上下文隔离* |
| [s05](./docs/zh/s05-skill-loading.md) | Skills | *按需加载, 而非预装* |
| [s06](./docs/zh/s06-context-compact.md) | Compact | *策略性遗忘* |
| [s07](./docs/zh/s07-task-system.md) | 任务系统 | *状态在压缩后存活* |
| [s08](./docs/zh/s08-background-tasks.md) | 后台任务 | *发射后不管* |
| [s09](./docs/zh/s09-agent-teams.md) | 智能体团队 | *追加即发送, 排空即读取* |
| [s10](./docs/zh/s10-team-protocols.md) | 团队协议 | *同一个 request_id, 两个协议* |
| [s11](./docs/zh/s11-autonomous-agents.md) | 自治智能体 | *轮询, 认领, 工作, 重复* |
| [s12](./docs/zh/s12-worktree-task-isolation.md) | Worktree + 任务隔离 | *目录隔离, 任务 ID 协调* |

## 许可证

MIT

---

**模型就是智能体。我们的工作就是给它工具, 然后让开。**
