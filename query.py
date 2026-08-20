"""查询流水线（MVP 可运行）：混合检索 → 管理 Agent 派发 → 输出。

运行：python pipelines/query.py "某商圈 LED 大屏适合哪些行业投放？"
（首次会自动调用 ingest 构建索引）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.embedding.client import get_embedder
from src.storage.vector_store import get_vector_store
from src.retrieval.hybrid import HybridRetriever
from src.agents.runtime import ManagerAgent, KnowledgeAgent


def main():
    embedder = get_embedder()
    vs = get_vector_store()
    retriever = HybridRetriever(vs, embedder)
    manager = ManagerAgent()
    manager.register(KnowledgeAgent(retriever))

    query = sys.argv[1] if len(sys.argv) > 1 else "某商圈 LED 大屏适合哪些行业投放？"

    # 若向量库为空（首次），先跑入库
    if not hasattr(vs, "_ids") or not vs._ids:
        print("[query] 向量库为空，先执行 ingest ...")
        import runpy
        runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipelines", "ingest.py"), run_name="__main__")

    print("\n=== 混合检索 Top-K ===")
    hits = retriever.retrieve(query, top_k=CFG.top_k)
    for i, h in enumerate(hits):
        print(f"[{i+1}] (score={h['score']:.3f}) {h['content'][:80]}...")

    print("\n=== A2A 管理 Agent 响应 ===")
    print(manager.receive_goal(query))


if __name__ == "__main__":
    main()
