This project contains a Go command-line tool for generating facet vector
embeddings from the set of indexed content on itch.io. It uses a Singular Value
Decomposition (SVD) based approach on a tag co-occurrence matrix.

On itch.io, every indexed piece of content is given a series of *facet* tags
that represent how creator has classified the content. As an example, the
browse category URL
[games/input-mouse/made-with-unity/tag-retro](https://itch.io/games/input-mouse/made-with-unity/tag-retro)
maps to the array of tags: `[c.1, in.2, tl.3, tg.retro]`.

By looking at the co-occurrence of tags on project pages in aggregate, a vector
space can be computed that represents the semantic relationship between tags.
(eg. A tag of `christmas` and `santa` might be used togther frequently on
single project pagess, so the computed vector space will have those two tags
near each other)

Any group of content can be summed into a set of tags can be reduced to a
single vector that can then be searched using a nearest neighbor algorithm to
find related content. These vectors know as **embeddings**.

### The Algorithm

1.  **Build a Co-occurrence Matrix**: The algorithm first constructs a large, symmetric matrix where rows and columns represent the unique tags from the vocabulary. A cell at `(row_i, col_j)` accumulates a weighted count of the games where `tag_i` and `tag_j` appeared together. Each game's contribution is scaled by its `weighted_rating` via a logistic curve `0.25 + 0.75 * sigmoid((r - 2.5) / 2)` centered on the rating formula's neutral prior of 2.5 (unrated games are treated as exactly neutral, matching what `game_weighted_rating` yields with zero votes; confidently low-rated games fall below unrated, confidently high-rated games approach `1.0`), so junk pages carry less influence. Each contribution is also divided by the game's tag count so heavily-tagged pages contribute mass linearly rather than quadratically. When the PPMI transform is used, marginal probabilities are smoothed with a 0.75 exponent (Levy et al. 2015) to damp PMI's bias toward rare tags.
2.  **Apply SVD for Dimensionality Reduction**: The co-occurrence matrix is often very large and noisy. To distill the most significant patterns, **Singular Value Decomposition (SVD)** is used. SVD factorizes the matrix into three separate matrices, capturing its underlying structure. This step effectively reduces the dimensionality of the data, filtering out noise and retaining the strongest signals.
3.  **Extract Embeddings**: The final embedding for each tag is a dense vector derived from the SVD output. By taking the top `N` dimensions (e.g., 32 or 64), we get a low-dimensional representation that captures the essence of the tag's relationship with all other tags. Each tag is now represented by a point in an `N`-dimensional space.
4.  **Normalize, Weight, and Store**: The vectors are normalized to unit length, then scaled by a per-facet SIF pooling weight (`a / (a + p)`, where `p` is the fraction of games carrying the facet) before being saved to a database table. Because the weight is baked into each vector's magnitude, a plain average of a game's facet vectors downstream is automatically a frequency-weighted average: ultra-common facets (`m.free`, `p.web`, ...) contribute little, while rare distinctive tags dominate. The weights are purely statistical; any product policy (eg. guaranteeing a minimum influence for monetization or platform facets) belongs in the application layer, which can recover a facet's unit vector by dividing the stored vector by its `weight` column.


### Schema

The algorithm will read from a table that looks like:

```sql
CREATE TABLE games_search (
  game_id integer NOT NULL,
  facets tsvector NOT NULL,
  -- ... other columns ignored
);
```

See
[games_search.md](https://github.com/itchio/facetembeddings/blob/master/games_search.md)
for detailed reference of facets stored in `games_search`.

The tool reads `game_id` and `facets` columns in batches, where `facets`
contains tags as a PostgreSQL tsvector (e.g., `'c.1' 'in.2' 'tg.horror'
'tg.puzzle'`). Tags are extracted from the tsvector using PostgreSQL's
`tsvector_to_array()` function.

And will generate a table that looks like:

```sql
CREATE TABLE facet_embeddings (
  facet text PRIMARY KEY,
  dim int NOT NULL,
  frequency integer NOT NULL DEFAULT 0,
  weight double precision NOT NULL DEFAULT 1,
  vector double precision[] NOT NULL,
  last_trained_at timestamp without time zone NOT NULL DEFAULT now()
);
```

This output table stores one row per unique tag, where:
- `facet` - The facet name (e.g., "c.1", "tg.puzzle")
- `dim` - The dimensionality of the embedding vector set at time of generation
- `frequency` - Document frequency: the number of games the facet appeared on
  at training time
- `weight` - The SIF pooling weight applied to the vector. The stored vector's
  magnitude equals this value (direction is unit length before scaling), so
  the column exists to document how each vector was scaled and to allow
  recovering the unit vector if needed.
- `vector` - The embedding vector as a PostgreSQL array: unit length scaled by
  `weight`
- `last_trained_at` - Timestamp when the embedding was computed

The table is created automatically if it doesn't exist and is truncated on each run before inserting new embeddings.

Every run also records the exact command and corpus stats behind the
embeddings so they can be recovered later:

- Database runs set the table's `COMMENT` to the equivalent CLI invocation
  plus item/tag counts and timing. Because it's a table comment, it travels
  with `pg_dump`/.sql exports of the table (view it with `\dt+` or
  `obj_description('facet_embeddings'::regclass)`).
- CSV runs write a `<output-file>.meta.json` sidecar with the same
  information plus the full config as JSON.

## Build

```sh
go build .
```

## Usage

The tool is configured via command-line flags.

```sh
./facetembeddings [flags]
```

### Configuration Flags

| Flag | Description | Default |
| --- | --- | --- |
| `-table` | Database table to write embeddings into. | `facet_embeddings` |
| `-output-file` | If set, write embeddings to this CSV file instead of the database (columns: facet, dim, frequency, weight, vector, last_trained_at). | _empty_ |
| `-sif-a` | SIF pooling-weight parameter `a` in `a/(a+p)`. Output vectors are scaled by their facet's weight. Set to `0` or below to disable weighting (vectors stay unit length). Default was chosen against corpus data: at `0.01` the median game's heaviest facet contributes ~33% of pooled weight (vs 44% at `0.001`, with 10% of games above 80%), while stopword facets stay crushed (`m.free` ≈ 0.011). | `0.01` |
| `-ppmi-alpha` | PPMI context-distribution smoothing exponent. `1` reproduces classic (unsmoothed) PPMI. | `0.75` |
| `-quality-weight` | Weight each game's co-occurrence contribution by its `weighted_rating` via a logistic centered on the neutral prior 2.5 (unrated games get `0.625`; confidently-bad games approach `0.25`, confidently-good approach `1.0`). | `true` |
| `-per-game-norm` | Divide each game's pair increments by its tag count minus one, so total contributed mass is linear in tag count. | `true` |
| `-embedding-dim` | The dimensionality of the output vectors. | `256` |
| `-min-tag-frequency` | Minimum number of times a tag must appear across all games to be included in the vocabulary. | `5` |
| `-max-tags` | Maximum number of unique tags to generate embeddings for, sorted by frequency. `0` means unlimited. With the default `rsvd` factorization the cap is no longer needed for compute; `-min-tag-frequency` is the quality gate. | `0` |
| `-min-cooccurrence` | Minimum number of games a tag pair must co-occur on to keep its matrix entry. Counted on raw game counts (before quality weighting and per-game normalization), so the flag always means "at least N games". Helps prune noise. | `1` |
| `-matrix-type` | Matrix type: `cooc` (raw co-occurrence) or `ppmi` (positive PMI). | `ppmi` |
| `-factorization` | Factorization method: `rsvd` (randomized truncated SVD over a sparse matrix — minutes instead of hours, memory scales with distinct tag pairs instead of vocabulary squared; ~90% top-10 neighbor agreement with exact SVD, top ranks essentially identical), `svd` (exact dense), or `als`. | `rsvd` |
| `-als-iterations` | Max ALS iterations (used when `-factorization=als`). | `15` |
| `-als-lambda` | ALS regularization parameter (λ). | `0.1` |
| `-als-convergence` | ALS early-stop threshold for relative loss change. | `1e-4` |
| `-batch-size` | Number of game rows to fetch from the database in a single batch. | `20000` |
| `-neighbors` | Comma-separated facets to print the top-10 nearest neighbors for (by cosine) after training. Purely a log-output sanity check — has no effect on the generated embeddings. Pick tags whose neighborhoods you can judge at a glance. | _empty_ |
| `-creator-tokens` | Experimental: add a `uid.<user_id>` token to each game at training time (joined from the `games` table), giving creators with ≥`-min-tag-frequency` games a queryable style vector. Training-side only — the tokens are not in `games_search.facets`, so site-pooled game vectors never include them. Note this substantially reshapes tag geometry too (~50% of tag top-10 neighbor lists change), so train into a separate table for creator-similarity features rather than replacing the main tag embeddings. | `false` |
| `-creator-context-weight` | The vocabulary-neutral alternative to `-creator-tokens`: if > 0, appends one synthetic training context per creator containing their signature tags (tags recurring on ≥2 of their games, capped at 32), weighted by this value relative to a real game. Injects portfolio-level co-occurrence without adding any tokens — vocabulary, frequencies, and SIF weights are unchanged. Measured tag-geometry shift vs the noise floor of ~0.9 top-10 overlap: `0.25` → 0.86 overlap, `0.5` → 0.83. | `0` |

### Production build

The defaults are the production configuration, so a full rebuild is:

```sh
./facetembeddings \
  -neighbors="tg.horror,tg.roguelike,tg.visual-novel,tg.liminal-space,tg.farming,tl.14"
```

This trains the full vocabulary (all facets on ≥5 games, ~25k) at 256
dimensions with randomized SVD, quality-weighted smoothed-PPMI counts, and SIF
pooling weights baked into the vector magnitudes, writing to
`facet_embeddings`. It completes in a minute or two. Before shipping the
table, read the `neighbors of ...` lines in the log and confirm the lists
look sane (e.g. `tg.horror` → `tg.survival-horror`, `tg.psychological-horror`).

To move the result to another instance, export the table — the table `COMMENT`
recording the exact command travels with the dump:

```sh
pg_dump -U postgres --clean -t facet_embeddings itchio > facet_embeddings.sql
```

> Note: the vectors are scaled by SIF weights, which drive near-universal
> facets (`m.free`, `p.web`, `c.1`, ...) close to zero. Consumers that want a
> guaranteed soft nudge from those facet classes must re-lift them at pooling
> time using the `weight` column (divide the vector by `weight` to recover the
> unit direction, then rescale to a floor).

### Example

To generate 128-dimensional embeddings for all tags that appear at least 10 times and save them to the `tag_vectors` table:

```sh
PGDATABASE=itchio_development ./facetembeddings \
  -table="tag_vectors" \
  -embedding-dim=128 \
  -min-tag-frequency=10
```
