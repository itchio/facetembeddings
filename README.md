This project contains a Go command-line tool for generating facet vector
embeddings from the set of indexed content on itch.io.

On itch.io, every indexed piece of content is given a series of *facet* tags
that represent how the creator has classified the content. As an example, the
browse category URL
[games/input-mouse/made-with-unity/tag-retro](https://itch.io/games/input-mouse/made-with-unity/tag-retro)
maps to the array of tags: `[c.1, in.2, tl.3, tg.retro]`.

By looking at the co-occurrence of tags on project pages in aggregate, a vector
space can be computed that represents the semantic relationship between tags.
(eg. `christmas` and `santa` are frequently used together on project pages, so
the computed vector space will have those two tags near each other.)

The tags of any piece of content can then be summed into a single vector,
which can be searched with a nearest neighbor lookup to find related content.
These vectors are known as **embeddings**.

### The Algorithm

1. Count how often each pair of tags appears together on the same game, across
   all of `games_search`. Each game's contribution is weighted by its
   `weighted_rating` (so potential low quality pages have less influence) and
   divided by its tag count (so heavily tagged pages don't dominate).
2. Convert the counts to PPMI (positive pointwise mutual information), which
   scores how much more often two tags appear together than chance would
   predict. Marginals are smoothed to keep rare tags from getting inflated
   scores.
3. Factorize the matrix down to `-embedding-dim` dimensions with a
   randomized truncated SVD. Each tag's row of the result is its embedding.
4. Normalize each vector to unit length, then scale it by a frequency-based
   weight (`a/(a+p)` where `p` is the fraction of games carrying the facet).
   Because the weight is baked into the vector's magnitude, a plain average
   of a game's facet vectors is automatically a weighted average: common
   facets like `m.free` contribute almost nothing, distinctive tags dominate.
   The weight is also stored in its own column so a consumer can divide it
   back out to recover the unit vector.

### Schema

The algorithm will read from a table that looks like:

```sql
CREATE TABLE games_search (
  game_id integer NOT NULL,
  facets tsvector NOT NULL,
  weighted_rating double precision,
  -- ... other columns ignored
);
```

See
[games_search.md](https://github.com/itchio/facetembeddings/blob/master/games_search.md)
for detailed reference of facets stored in `games_search`.

The tool reads rows in batches, where `facets` contains tags as a PostgreSQL
tsvector (e.g., `'c.1' 'in.2' 'tg.horror' 'tg.puzzle'`), extracted with
`tsvector_to_array()`.

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

One row per tag, where:

- `facet` - The facet name (e.g., "c.1", "tg.puzzle")
- `dim` - The dimensionality of the embedding vector
- `frequency` - The number of games the facet appeared on at training time
- `weight` - The pooling weight applied to the vector (the vector's magnitude
  equals this value)
- `vector` - The embedding vector: unit length scaled by `weight`
- `last_trained_at` - Timestamp when the embedding was computed

The table is created automatically if it doesn't exist and is truncated on
each run before inserting new embeddings.

Each run records the command used to generate it: database runs set the
table's `COMMENT` (visible with `\dt+`, and included in `pg_dump` output),
CSV runs write a `<output-file>.meta.json` sidecar.

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
| `-embedding-dim` | The dimensionality of the output vectors. | `256` |
| `-min-tag-frequency` | Minimum number of games a tag must appear on to be included in the vocabulary. | `5` |
| `-max-tags` | Maximum number of tags to embed, kept by frequency. `0` means unlimited. | `0` |
| `-min-cooccurrence` | Minimum number of games a tag pair must appear on together to keep its matrix entry (counted before weighting). | `1` |
| `-sif-a` | Parameter `a` of the `a/(a+p)` pooling weight applied to output vectors. `0` disables the weighting (vectors stay unit length). | `0.01` |
| `-ppmi-alpha` | Smoothing exponent for PPMI marginals. `1` disables smoothing. | `0.75` |
| `-quality-weight` | Scale each game's contribution by its `weighted_rating`. Unrated games count `0.625`, badly rated less, well rated up to `1.0`. | `true` |
| `-per-game-norm` | Divide each game's contribution by its tag count. | `true` |
| `-matrix-type` | `ppmi` or `cooc` (raw co-occurrence counts). | `ppmi` |
| `-factorization` | `rsvd` (randomized truncated SVD, fast), `svd` (exact, slow on large vocabularies), or `als`. | `rsvd` |
| `-als-iterations` | Max ALS iterations. | `15` |
| `-als-lambda` | ALS regularization parameter (λ). | `0.1` |
| `-als-convergence` | ALS early-stop threshold for relative loss change. | `1e-4` |
| `-batch-size` | Number of game rows to fetch from the database per query. | `20000` |
| `-neighbors` | Comma-separated facets to print nearest neighbors for after training, as a sanity check. Doesn't affect the output. | _empty_ |

### Production build

The defaults are the production configuration, so a full rebuild is:

```sh
./facetembeddings \
  -neighbors="tg.horror,tg.roguelike,tg.visual-novel,tg.liminal-space,tg.farming"
```

This trains all facets appearing on 5+ games (~25k) and finishes in a minute
or two. Before shipping the table, check the `neighbors of ...` lines in the
log for anything that looks off.

To move the result to another instance, export the table:

```sh
pg_dump -U postgres --clean -t facet_embeddings itchio > facet_embeddings.sql
```

Note that the pooling weights push near-universal facets (`m.free`, `p.web`,
`c.1`, ...) close to zero. A consumer that wants those facets to still
influence similarity needs to rescale them using the `weight` column.

### Example

To generate 128-dimensional embeddings for all tags that appear at least 10
times and save them to the `tag_vectors` table:

```sh
PGDATABASE=itchio_development ./facetembeddings \
  -table="tag_vectors" \
  -embedding-dim=128 \
  -min-tag-frequency=10
```
