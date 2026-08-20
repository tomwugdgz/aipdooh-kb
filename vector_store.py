"""向量存储：SimpleVectorStore（numpy 内存，MVP 零依赖）与 MilvusVectorStore（生产分布式）。"""
from __future__ import annotations
import json
import os
import numpy as np
from src.config import CFG


class VectorStore:
    def upsert(self, ids, vectors, payloads): ...
    def search(self, vector, top_k: int, filters: dict | None = None) -> list[dict]: ...


class SimpleVectorStore(VectorStore):
    """内存向量库：余弦相似度 + 磁盘持久化（npz + json）。演示与冒烟测试用。"""

    def __init__(self, path: str = "kb_store.npz"):
        self.path = path
        self._ids: list[str] = []
        self._vecs: list[np.ndarray] = []
        self._payloads: list[dict] = []
        self._corpus: list[str] = []
        self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return
        try:
            data = np.load(self.path, allow_pickle=False)
            self._vecs = [data[f"v{i}"] for i in range(int(data["n"]))]
            with open(self.path + ".json", "r", encoding="utf-8") as f:
                blob = json.load(f)
            self._ids = blob["ids"]
            self._payloads = blob["payloads"]
            self._corpus = blob["corpus"]
        except Exception:
            pass

    def _save(self):
        if self._vecs:
            arr = np.stack(self._vecs)
            np.savez(self.path, n=np.array(len(self._vecs)), **{f"v{i}": self._vecs[i] for i in range(len(self._vecs))})
            with open(self.path + ".json", "w", encoding="utf-8") as f:
                json.dump({"ids": self._ids, "payloads": self._payloads, "corpus": self._corpus}, f, ensure_ascii=False)

    def upsert(self, ids, vectors, payloads):
        for i, v, p in zip(ids, vectors, payloads):
            self._ids.append(i)
            self._vecs.append(np.asarray(v, dtype=np.float32))
            self._payloads.append(p)
            self._corpus.append(p.get("content", ""))
        self._save()

    def search(self, vector, top_k: int = CFG.top_k, filters: dict | None = None) -> list[dict]:
        if not self._vecs:
            return []
        q = np.asarray(vector, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        mat = np.vstack(self._vecs)
        norms = np.linalg.norm(mat, axis=1)
        norms[norms == 0] = 1e-9
        sims = mat @ q / norms
        order = np.argsort(-sims)
        results = []
        for idx in order:
            p = self._payloads[idx]
            if filters and not _match(p.get("acl", {}), filters):
                continue
            results.append({
                "id": self._ids[idx],
                "score": float(sims[idx]),
                "content": p.get("content", ""),
                "meta": p.get("meta", {}),
                "acl": p.get("acl", {}),
            })
            if len(results) >= top_k:
                break
        return results


def _match(acl: dict, filters: dict) -> bool:
    """ACL 过滤：tenant_id 必须一致（演示级）。"""
    if "tenant_id" in filters and acl.get("tenant_id") != filters["tenant_id"]:
        return False
    return True


class MilvusVectorStore(VectorStore):
    """生产向量库：Milvus 分布式，十亿级 chunk、分片、副本、冷热分层。"""

    def __init__(self, uri: str = CFG.milvus_uri, collection: str = CFG.collection_name):
        from pymilvus import MilvusClient  # 需 pip install pymilvus
        self.client = MilvusClient(uri)
        self.collection = collection

    def upsert(self, ids, vectors, payloads):
        data = [
            {"id": i, "vector": v.tolist() if hasattr(v, "tolist") else v, "payload": p}
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(self.collection, data)

    def search(self, vector, top_k: int = CFG.top_k, filters: dict | None = None) -> list[dict]:
        expr = ""
        if filters and filters.get("tenant_id"):
            expr = f"payload['acl']['tenant_id'] == '{filters['tenant_id']}'"
        res = self.client.search(
            self.collection, [vector.tolist()], limit=top_k, filter=expr
        )
        return [
            {"id": r["id"], "score": r["distance"], "content": r["entity"].get("payload", {}).get("content", "")}
            for r in res[0]
        ]


def get_vector_store() -> VectorStore:
    if CFG.vector_backend == "milvus":
        return MilvusVectorStore()
    return SimpleVectorStore(path=os.environ.get("KB_STORE_PATH", "kb_store.npz"))
