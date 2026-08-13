# Sink

**Scenario:** You want traces on disk for local debugging, in SQLite for queries, and optionally exported to your OpenTelemetry collector — without changing call sites.

A **Sink** is where traces go.

| Sink | Use |
| --- | --- |
| `JSONLSink` | Default; append-only, easy to ship |
| `SQLiteSink` | Query by environment / id; supports late labels |
| `OTelSink` | Export `gen_ai.*` spans (`pip install enroute[otel]`) |
| `MultiSink` | Fan out to several sinks |

Redaction and sampling run **before** the sink, so PII never reaches disk when configured correctly. Writes are queued on a background thread so tracing does not block requests — call `client.flush()` or `client.close()` on shutdown.
