# SeaGlass AI Dataset Plan

## Scope and principles

SeaGlass should be trained on carefully licensed, high-signal text. The
eventual mixture targets approximately **20% Roblox-related material** and
approximately **80% broad general-capability material**. The Roblox portion
emphasizes explaining systems, reasoning about trade-offs, and writing or
reviewing Luau—not game-popularity chatter. This is a future data-curation plan
only: it does not download, generate, or train on a dataset.

Every candidate source must have a clear right to use it for the intended
training purpose. Preserve source, license, collection date, and quality score
in data provenance records. Exclude private conversations, personal data, and
content with uncertain permissions.

## Priority domains

### Roblox science and engineering — approximately 20% of the mixture

Favor authoritative Roblox documentation, API references, Creator Hub learning
material where licensing permits use, carefully selected public technical
articles, and high-quality open-source Luau examples with compatible licenses.
The corpus should prioritize explanations and examples involving:

- Luau language features, types, modules, debugging, testing, and code design.
- Roblox APIs, services, instances, events, and engine systems.
- Client/server responsibilities, RemoteEvents/RemoteFunctions, replication,
  ownership, validation, security, and networking trade-offs.
- Physics, assemblies, constraints, collisions, raycasting, forces, and
  simulation stability.
- Character movement, humanoids, controllers, animation state, IK, and rigs.
- Rendering, lighting, materials, post-processing, terrain, and asset behavior.
- Pathfinding, navigation, steering, NPC architecture, and game AI.
- Profiling, frame-time analysis, memory behavior, performance budgets,
  streaming, batching, and optimization techniques.
- Engine architecture and transferable simulation or real-time systems concepts.

Prefer content that explains *why* a system behaves as it does, includes
constraints or failure modes, and distinguishes verified facts from advice.

### Broad general capability — approximately 80% of the mixture

Build broad capability with carefully selected material spanning general
knowledge, reasoning and problem solving, coding, conversation/dialogue,
mathematics, science, stories, and creative writing. Include general software
engineering, Python, systems programming, networking, graphics, physics,
machine learning, and game-engine design as useful technical foundations.
This balance helps SeaGlass reason across concepts rather than memorizing API
fragments, while stories and dialogue improve flexible communication.

## Content to deprioritize or exclude

Do not use low-signal material merely because it mentions Roblox. Strongly
deprioritize:

- Listicles, clickbait, and “top Roblox games” pages.
- Repetitive game descriptions, reviews, gameplay chatter, and popularity news.
- SEO spam, scraped pages, keyword stuffing, and duplicate syndications.
- Repetitive update/news posts without lasting technical explanation.
- Unverified tutorials that encourage insecure remote handling or cargo-cult
  code, unless retained as clearly labeled negative/review examples.

## Curation workflow

1. Collect only approved, attributable candidate sources.
2. Normalize text, remove boilerplate/navigation, and retain provenance.
3. Deduplicate exact and near-duplicate documents.
4. Score documents for technical depth, clarity, correctness, license status,
   and domain fit.
5. Filter for the priority topics above; reject low-quality or unsafe material.
6. Split by source/document before train-validation-test partitioning to reduce
   leakage.
7. Create evaluation sets for Luau/API reasoning, client/server safety,
   replication, physics, optimization, and technical explanation quality.

## Dataset boundaries

The eventual corpus must not include the user's personal conversation history,
private messages, credentials, or other personal information. Persona and tone
are specified separately in `PERSONA_SPEC.md`; they should not be learned by
mixing private conversation data into technical knowledge.
