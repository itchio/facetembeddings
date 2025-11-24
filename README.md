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

1.  **Build a Co-occurrence Matrix**: The algorithm first constructs a large, symmetric matrix where rows and columns represent the unique tags from the vocabulary. A cell at `(row_i, col_j)` stores the number of times `tag_i` and `tag_j` appeared together on the same game. This matrix captures the raw co-occurrence relationship between all pairs of tags.
2.  **Apply SVD for Dimensionality Reduction**: The co-occurrence matrix is often very large and noisy. To distill the most significant patterns, **Singular Value Decomposition (SVD)** is used. SVD factorizes the matrix into three separate matrices, capturing its underlying structure. This step effectively reduces the dimensionality of the data, filtering out noise and retaining the strongest signals.
3.  **Extract Embeddings**: The final embedding for each tag is a dense vector derived from the SVD output. By taking the top `N` dimensions (e.g., 32 or 64), we get a low-dimensional representation that captures the essence of the tag's relationship with all other tags. Each tag is now represented by a point in an `N`-dimensional space.
4.  **Normalize and Store**: The vectors are normalized to unit length and saved to a database table for downstream use.


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
  vector double precision[] NOT NULL,
  last_trained_at timestamp without time zone NOT NULL DEFAULT now()
);
```

This output table stores one row per unique tag, where:
- `facet` - The facet name (e.g., "c.1", "tg.puzzle")
- `dim` - The dimensionality of the embedding vector set at time of generation
- `vector` - The normalized embedding vector as a PostgreSQL array
- `last_trained_at` - Timestamp when the embedding was computed

The table is created automatically if it doesn't exist and is truncated on each run before inserting new embeddings.

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
| `-output-file` | If set, write embeddings to this CSV file instead of the database (columns: facet, dim, vector, last_trained_at). | _empty_ |
| `-embedding-dim` | The dimensionality of the output vectors. | `32` |
| `-min-tag-frequency` | Minimum number of times a tag must appear across all games to be included in the vocabulary. | `5` |
| `-max-tags` | Maximum number of unique tags to generate embeddings for, sorted by frequency. `0` means unlimited. | `20000` |
| `-min-cooccurrence` | Minimum co-occurrence count required to keep an entry in the matrix. Helps prune noise. | `1` |
| `-matrix-type` | Matrix type: `cooc` (raw co-occurrence) or `ppmi` (positive PMI). | `ppmi` |
| `-factorization` | Factorization method: `svd` or `als`. | `svd` |
| `-als-iterations` | Max ALS iterations (used when `-factorization=als`). | `15` |
| `-als-lambda` | ALS regularization parameter (λ). | `0.1` |
| `-als-convergence` | ALS early-stop threshold for relative loss change. | `1e-4` |
| `-batch-size` | Number of game rows to fetch from the database in a single batch. | `20000` |

### Example

To generate 128-dimensional embeddings for all tags that appear at least 10 times and save them to the `tag_vectors` table:

```sh
PGDATABASE=itchio_development ./facetembeddings \
  -table="tag_vectors" \
  -embedding-dim=128 \
  -min-tag-frequency=10
```
