# AIpDOOH 户外媒体知识库 + A2A 智能体 · 底层框架脚手架

为 **AIpDOOH（AI 化程序化户外媒体）** 搭建的知识底座雏形：沉淀全量户外媒体案例 + 投放数据 + 公司知识；
数据全量留在本地；上层由 **A2A 智能体系统**（管理 Agent 统筹领域 Agent）驱动。

## 设计要点（对应你的约束）
- **千亿级原始数据** → 走湖仓（MinIO + Iceberg + ClickHouse），不进向量库。
- **十亿级知识单元** → 分布式向量（Milvus）+ 图谱（NebulaGraph），分片 + 冷热分层。
- **全本地** → LLM/Embedding/Rerank 均私有化（vLLM + bge），零出域。
- **A2A** → Google A2A 协议（AgentCard/Task）做互操作 + 自研运行时做编排。

## 快速开始（零依赖，任何机器可跑）
```bash
cd aipdooh-kb
pip install numpy
python pipelines/ingest.py                       # 写入示例户外媒体案例
python pipelines/query.py "某商圈 LED 大屏适合哪些行业投放？"
```
`ingest` 把示例案例分块、向量化、写入本地向量库（numpy 持久化）；
`query` 走「向量 + BM25 混合召回 → RRF 融合 → 管理 Agent 派发 → 知识 Agent 回答」。

## 目录结构
```
aipdooh-kb/
├── docker-compose.yml        # 生产基础设施：Milvus/PG/MinIO/Redis（vLLM 需 GPU）
├── requirements.txt
├── src/
│   ├── config.py             # 全局配置（向量库/嵌入/分块/ACL 开关）
│   ├── connectors/           # 数据源连接器：媒体案例/投放日志/公司知识
│   ├── processing/           # 清洗 + 语义分块
│   ├── embedding/            # MockEmbedder(演示) / BgeEmbedder(生产)
│   ├── storage/              # SimpleVectorStore(内存) / MilvusVectorStore / 元数据库
│   ├── retrieval/            # 混合检索 + RRF + Rerank(可插拔)
│   ├── agents/               # A2A 运行时：管理Agent + 知识Agent + 规划Agent
│   └── service/              # FastAPI 网关（/search /chat /agent/task /agent.json）
└── pipelines/
    ├── ingest.py             # 入库流水线
    └── query.py              # 查询流水线
```

## 切到生产
1. `config.py` 设 `vector_backend=milvus`、`embedder=bge`、`meta_backend=postgres`。
2. `docker compose up -d` 起基础设施；GPU 机器启用 vLLM 跑 LLM/Embedding/Rerank。
3. 把 `connectors/pdooh_sources.py` 的 `pull()` 换成真实 JDBC/API/Kafka 读取。
4. 在 `agents/runtime.py` 注册更多领域 Agent（投放优化、创意、数据、合规）。

## 当前为 MVP 雏形
- 向量/嵌入用 numpy + 哈希向量演示，保证可运行；生产替换为 Milvus + bge-m3。
- 管理 Agent 用规则拆解；生产接 LLM 做目标分解与编排。
- 图谱（NebulaGraph）与湖仓（Iceberg/ClickHouse）为架构预留，未在 MVP 实现。
