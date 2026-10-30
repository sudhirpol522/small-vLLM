# Writing and maintaining the walkthroughs

[Guide index](README.md) · [Source index](api-index.md) · [Review notes](review-notes.md)

## Documentation layout

The repository README introduces the project, shows the component map, and directs readers to the guide. The guide index provides the reading order. Concept chapters explain why an optimization exists. Component chapters explain how the code implements it. The source index tracks coverage at the definition level.

Keep full method explanations out of the repository README. A reader should be able to find the execution path and choose a chapter before encountering allocation arithmetic or attention tensor details.

## Class introduction

For each class, explain its responsibility, the state it owns, and which components call it. Link to the source definition. Identify important invariants before explaining functions that rely on them.

A diagram is useful when it explains relationships, ownership, or a state transition. Reuse existing block IDs and request names across a chapter so the reader can follow one example.

## Method template

Use the following order, adjusting the length to the method:

1. State the purpose in one or two sentences.
2. Link to the source and show the relevant snippet. Preserve behavior when condensing comments.
3. Define the inputs and the state the method reads.
4. Follow a concrete example through the logical sections of the code.
5. Show the resulting state and return value.
6. Explain meaningful branches, invariants, and limitations visible in the implementation.
7. Connect to the next method in the request's execution path.

For a property such as `num_free`, a few sentences are enough. For `allocate` or `schedule`, use multiple snippets and before/after tables. Do not repeat the entire function and every individual line unless the repetition helps the explanation.

## Source and coverage rules

Use commit-pinned GitHub links for audited examples. When the source changes, update snippets, line links, worked examples, and coverage status together. Treat comments as useful context and check their claims against the code.

Use `Draft walkthrough` when an explanation exists but still needs review. Use `Planned` for an inventoried symbol without a method walkthrough. Mark a method reviewed only after checking behavior, arithmetic, and its caller/callee assumptions. An architecture mention alone does not count as a method walkthrough.

The source index includes explicitly declared classes, functions, methods, properties, and nested helpers. Lambdas are explained with their containing code. Dataclass-generated methods and code embedded in transformation strings are not independent source definitions in this index.

## Diagram conventions

Use Mermaid for component maps and state transitions. Use the detailed supplied diagrams for physical block layouts, queues, and before/after examples. Keep images under `docs/assets` with descriptive names and include a caption and useful alt text.

Distinguish logical block positions from physical block IDs, block ownership from cached contents, and registered metadata from completed model computation. Show shared data with multiple references to the same physical block.

Diagram 7 is still awaiting an image. The existing [review notes](review-notes.md) record changes needed to some supplied figures. Complete those revisions before calling the documentation publication-ready.

## Applying this starter

The documentation ZIP is an overlay for the existing repository: place `README.md` at the repository root and `docs/` beside `serve.py`. It contains documentation and image assets; it does not replace the implementation files. Relative source links in the root README resolve against those existing files.

Review the draft chapters and image corrections in a documentation branch. Once each new chapter is written, add its link to the reading order and update its definitions in the coverage index.
