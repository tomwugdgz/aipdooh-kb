"""A2A 智能体层：管理 Agent（AI 管理 A2A）+ 领域 Agent（知识/规划/优化...）。

遵循 Google A2A 思路：每个 Agent 暴露 AgentCard（能力/端点/鉴权），通过 Task 通信。
内部用自研运行时做目标拆解与调度；生产把 send_task 换成 HTTP(JSON-RPC) 调用远端 AgentCard。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class AgentCard:
    card_id: str
    name: str
    endpoint: str
    capabilities: list[str]
    auth: str = "internal"


@dataclass
class Task:
    task_id: str
    agent: str
    goal: str
    parent_id: str | None = None
    status: str = "pending"
    result: str | None = None


class Agent:
    def __init__(self, card: AgentCard):
        self.card = card

    def handle(self, task: Task) -> str:
        raise NotImplementedError


class KnowledgeAgent(Agent):
    """知识 Agent：调用 RAG 检索回答（对接 HybridRetriever）。"""

    def __init__(self, retriever):
        super().__init__(AgentCard("agent.knowledge", "知识Agent",
                                    "http://localhost:8081", ["rag", "search"]))
        self.retriever = retriever

    def handle(self, task: Task) -> str:
        hits = self.retriever.retrieve(task.goal, top_k=3)
        ctx = "\n".join(f"[{i+1}] {h['content']}" for i, h in enumerate(hits))
        return f"[知识Agent] 基于检索上下文回答：\n{ctx}"


class MediaPlanningAgent(Agent):
    """媒体规划 Agent（占位）：基于案例与数据做点位/商圈规划。"""

    def __init__(self):
        super().__init__(AgentCard("agent.planning", "媒体规划Agent",
                                    "http://localhost:8082", ["planning"]))

    def handle(self, task: Task) -> str:
        return f"[媒体规划Agent] 已接收规划目标：{task.goal}（生产接知识+数据 Agent 协同）"


class ManagerAgent(Agent):
    """管理 Agent：接收高层目标 → 拆解 → 经 A2A 派发 → 聚合 → 护栏。"""

    def __init__(self):
        super().__init__(AgentCard("agent.manager", "管理Agent",
                                    "http://localhost:8080", ["orchestrate"]))
        self.registry: dict[str, Agent] = {}
        self.tasks: list[Task] = []

    def register(self, agent: Agent):
        self.registry[agent.card.card_id] = agent

    def send_task(self, agent_id: str, goal: str, parent_id: str | None = None) -> Task:
        """A2A Task 派发（生产改为 HTTP POST AgentCard.endpoint/a2a/tasks）。"""
        task = Task(task_id=f"t{len(self.tasks)+1}", agent=agent_id, goal=goal, parent_id=parent_id)
        agent = self.registry.get(agent_id)
        if not agent:
            task.status = "failed"
            task.result = f"未知 Agent: {agent_id}"
            return task
        task.result = agent.handle(task)
        task.status = "completed"
        self.tasks.append(task)
        return task

    def receive_goal(self, goal: str) -> str:
        """极简编排：含'规划'→媒体规划Agent；否则→知识Agent。生产用 LLM 做拆解。"""
        if any(k in goal for k in ["规划", "选点", "投放方案"]):
            t = self.send_task("agent.planning", goal)
        else:
            t = self.send_task("agent.knowledge", goal)
        return f"[管理Agent] 目标：{goal}\n→ {t.result}"
