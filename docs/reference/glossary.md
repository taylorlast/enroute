# Glossary

| Term | Definition |
| --- | --- |
| **Trace** | Ordered record of one interaction or episode, plus an optional outcome. |
| **Step** | One entry in a trace: a `Decision`, `LLMCall`, `ToolCall`, or `Event`. |
| **Decision** | One model turn: observation, model context/output, parsed action, tool results, reward events. |
| **Credit** | Late reward attached to an episode or one decision (`trace.credit`). |
| **Return** | Discounted sum of future rewards from a decision (`trace.returns`). How research inherits credit from a later outcome. |
| **Outcome** | Labels, scores, reward, and feedback attached to a trace. |
| **Sink** | Destination that persists traces (JSONL, SQLite, OTel, …). |
| **Environment** | Versioned simulator (`class WordleEnv(Environment[Obs, State])`) with instructions, tools, and scorers that produces scored traces. |
| **Observation** | What the agent may see (`Observation` / subclass). Built by the `observe` hook; returned by `reset` / `step`. |
| **State** | Internal episode data on `env.state`. May include hidden fields. |
| **Task** | One case an environment can run (`TaskData`). Seeds the env via `setup`; may carry a goal. |
| **Rollout** | One execution of a task through an environment. Synonym: **episode**. |
| **Episode** | Gymnasium name for a rollout. One episode emits one Trace. |
| **Fingerprint** | Hash of an environment's tools, instructions, and scorers. Compatibility key stored on the trace. |
| **Dataset** | Named, content-hashed collection of traces for benchmarks/training. |
| **Benchmark** | Matrix run of an environment across models producing a `Report`. |
| **Routing policy** | Object that orders model/provider routes for a request. |
| **Provider** | Adapter that speaks a vendor API and returns normalized types. |
| **Catalog** | Offline model metadata (context, pricing, modalities). |
