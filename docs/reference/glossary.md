# Glossary

| Term | Definition |
| --- | --- |
| **Trace** | Ordered record of LLM calls, tool calls, and events for one interaction, plus an optional outcome. |
| **Step** | One entry in a trace: an `LLMCall`, `ToolCall`, or `Event`. |
| **Outcome** | Labels, scores, reward, and feedback attached to a trace. |
| **Sink** | Destination that persists traces (JSONL, SQLite, OTel, …). |
| **Environment** | Versioned harness of tasks, tools, and scorers that produces scored traces. |
| **Task** | One case an environment can run (`TaskData`). |
| **Rollout** | One execution of a task through an environment. |
| **Dataset** | Named, content-hashed collection of traces for benchmarks/training. |
| **Benchmark** | Matrix run of an environment across models producing a `Report`. |
| **Routing policy** | Object that orders model/provider routes for a request. |
| **Provider** | Adapter that speaks a vendor API and returns normalized types. |
| **Catalog** | Offline model metadata (context, pricing, modalities). |
