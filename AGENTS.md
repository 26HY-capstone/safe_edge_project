# Vision Guard project instructions

- Before planning, designing, or implementing project features, read `docs/VISION_GUARD_SPEC.md` and treat it as the primary functional and architectural specification.
- Keep `docs/VISION_GUARD_SPEC.md` as a living document. Update it when the user changes requirements, architecture, interfaces, priorities, MVP scope, or Edge constraints.
- Do not silently change the specification. Make requirement changes explicit and keep implementation consistent with the latest documented version.
- For every new feature, determine its module, inputs, outputs, interfaces, Edge compute cost, and whether it should run continuously or by trigger.
- Prefer designs that can run in real time on NVIDIA Jetson Edge devices.
