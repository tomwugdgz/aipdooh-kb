"""入库流水线（MVP 可运行）：连接器拉取 → 加工 → 向量化 → 写入向量库 + 元数据。

运行：python pipelines/ingest.py
生产：把 vector_backend=milvus / embedder=bge 写入 config，接真实连接器即可。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.embedding.client import get_embedder
from src.storage.vector_store import get_vector_store
from src.storage.meta_store import get_meta_store
from src.connectors.pdooh_sources import all_connectors
from src.processing.pipeline import process_doc


def main():
    embedder = get_embedder()
    vs = get_vector_store()
    meta = get_meta_store()

    all_chunks = []
    for conn in all_connectors():
        for raw in conn.pull():
            meta.add_document({
                "id": raw.doc_id, "tenant_id": raw.tenant_id, "source": raw.source,
                "doc_type": raw.doc_type, "title": raw.title, "status": "active", "version": 1,
            })
            chunks = process_doc(raw, embedder)
            all_chunks.extend(chunks)

    # 批量向量化
    ids = [f"{c['doc_id']}#{c['idx']}" for c in all_chunks]
    vecs = embedder.embed_batch([c["content"] for c in all_chunks])
    payloads = [{"content": c["content"], "meta": c["meta"], "acl": c["acl"]} for c in all_chunks]
    vs.upsert(ids, vecs, payloads)

    for c, cid in zip(all_chunks, ids):
        meta.add_chunk({"id": cid, "doc_id": c["doc_id"], "tenant_id": c["tenant_id"],
                        "idx": c["idx"], "content": c["content"], "meta": c["meta"], "acl": c["acl"]})

    # 混合召回所需的 BM25 语料随向量库持久化（retriever 检索时自动构建）

    print(f"[ingest] 完成：文档 {len(all_chunks)} 块，向量库={CFG.vector_backend}，嵌入={CFG.embedder}")


if __name__ == "__main__":
    main()
