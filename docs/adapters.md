# Adapters

Adapters provide concrete implementations behind the `Session` workflow.

Current adapter areas include:

- validation
- enrichment
- extract
- aggregation
- dataops

The `Session` can use default adapters where available, or you can register a
custom adapter with `session.register_adapter(...)`. Registered adapters only
apply to the session they were registered on.
