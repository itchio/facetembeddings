package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/lib/pq"
)

var dbConnString string

func init() {
	dbName := os.Getenv("PGDATABASE")
	if dbName == "" {
		dbName = "itchio"
	}
	dbConnString = fmt.Sprintf("user=postgres dbname=%s sslmode=disable", dbName)
	log.Printf("Database: %s", dbConnString)
}

// ItemTags represents the tags attached to a single game.
type ItemTags struct {
	GameID int64
	Tags   []string

	// Weight scales this item's contribution to co-occurrence counts
	// (quality weighting). 1 when quality weighting is disabled.
	Weight float64
}

// shouldFilterTag returns true if the tag should be excluded from embeddings.
// Filters out negative boolean facets that have low information content.
func shouldFilterTag(tag string) bool {
	switch tag {
	case "n.no":  // Not NSFW (vast majority, low signal)
		return true
	case "j.no":  // Not a game jam (common, low signal)
		return true
	default:
		return false
	}
}

const selectTagsQuery = `
SELECT game_id, tsvector_to_array(facets) AS tags, weighted_rating
FROM games_search
WHERE facets IS NOT NULL AND game_id > $1
ORDER BY game_id
LIMIT $2
`

// itemQualityWeight maps a game's weighted_rating into a co-occurrence
// contribution weight in (0.25, 1.0). weighted_rating (game_weighted_rating
// in the site schema) shrinks the 0-5 star average toward the neutral prior
// 2.5 when votes are few, then adds log(count)*(average-2.5)/2.5 — so values
// far below 2.5 mean confidently bad (not "few votes") and the scale is
// unbounded upward. Unrated games (NULL, or the games_search column's 0
// default) are treated as exactly neutral, matching what the formula itself
// yields at count=0; a logistic centered there dampens confidently-bad games
// below unrated ones and boosts confidently-good ones toward 1.
func itemQualityWeight(rating sql.NullFloat64) float64 {
	const neutral = 2.5 // the rating formula's own prior
	const scale = 2.0   // logistic width, roughly the corpus spread around neutral

	r := neutral
	if rating.Valid && rating.Float64 != 0 {
		r = rating.Float64
	}
	return 0.25 + 0.75/(1+math.Exp(-(r-neutral)/scale))
}

func Connect(ctx context.Context) (*sql.DB, error) {
	if dbConnString == "" {
		return nil, errors.New("database connection string not initialized")
	}

	db, err := sql.Open("postgres", dbConnString)
	if err != nil {
		return nil, fmt.Errorf("open postgres: %w", err)
	}

	if err := db.PingContext(ctx); err != nil {
		db.Close()
		return nil, fmt.Errorf("ping postgres: %w", err)
	}

	return db, nil
}

// streams game/tag rows ordered by game_id in batches. When qualityWeight is
// set, each item's co-occurrence contribution is scaled by its weighted_rating
// (see itemQualityWeight); otherwise every item weighs 1.
func LoadItemTags(ctx context.Context, db *sql.DB, batchSize int, qualityWeight bool) ([]ItemTags, error) {
	if batchSize <= 0 {
		batchSize = 10_000
	}

	var items []ItemTags
	lastID := int64(-1)
	filteredCount := 0

	for {
		rows, err := db.QueryContext(ctx, selectTagsQuery, lastID, batchSize)
		if err != nil {
			return nil, fmt.Errorf("query item tags: %w", err)
		}

		rowCount := 0
		for rows.Next() {
			rowCount++

			var item ItemTags
			var tagArray pq.StringArray
			var rating sql.NullFloat64
			if err := rows.Scan(&item.GameID, &tagArray, &rating); err != nil {
				rows.Close()
				return nil, fmt.Errorf("scan item tags: %w", err)
			}

			item.Weight = 1
			if qualityWeight {
				item.Weight = itemQualityWeight(rating)
			}

			// Filter out negative boolean tags
			for _, tag := range tagArray {
				if !shouldFilterTag(tag) {
					item.Tags = append(item.Tags, tag)
				} else {
					filteredCount++
				}
			}

			if item.GameID > lastID {
				lastID = item.GameID
			}

			if len(item.Tags) == 0 {
				continue
			}

			items = append(items, item)
		}

		if err := rows.Err(); err != nil {
			rows.Close()
			return nil, fmt.Errorf("iterate item tags: %w", err)
		}
		rows.Close()

		if rowCount < batchSize {
			break
		}
	}

	if filteredCount > 0 {
		log.Printf("filtered out %d negative boolean tags (n.no, j.no)", filteredCount)
	}

	return items, nil
}

// truncates and repopulates the facet embeddings table with the resulting
// vectors. freq holds the per-facet document frequency (number of games the
// facet appeared on at training time); weights holds the SIF pooling weight
// already baked into each stored vector's magnitude. Both are stored so the
// table documents how each vector was scaled.
func SaveEmbeddings(ctx context.Context, db *sql.DB, table string, embeddings map[string][]float64, freq map[string]int, weights map[string]float64, dim int) (int, error) {
	if err := ensureEmbeddingsTable(ctx, db, table); err != nil {
		return 0, fmt.Errorf("ensure tag_embeddings: %w", err)
	}

	tx, err := db.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("begin tx: %w", err)
	}
	defer func() {
		_ = tx.Rollback()
	}()

	truncateSQL := fmt.Sprintf(`TRUNCATE %s`, quoteIdentifier(table))
	if _, err := tx.ExecContext(ctx, truncateSQL); err != nil {
		return 0, fmt.Errorf("truncate tag_embeddings: %w", err)
	}

	insertSQL := fmt.Sprintf(`
INSERT INTO %s (facet, dim, frequency, weight, vector, last_trained_at)
VALUES ($1, $2, $3, $4, $5, now())
`, quoteIdentifier(table))

	tags := make([]string, 0, len(embeddings))
	for tag := range embeddings {
		tags = append(tags, tag)
	}
	sort.Strings(tags)

	for _, tag := range tags {
		vec := embeddings[tag]
		weight := 1.0
		if w, ok := weights[tag]; ok {
			weight = w
		}
		if _, err := tx.ExecContext(ctx, insertSQL, tag, dim, freq[tag], weight, pq.Array(vec)); err != nil {
			return 0, fmt.Errorf("insert embedding for %s: %w", tag, err)
		}
	}

	if err := tx.Commit(); err != nil {
		return 0, fmt.Errorf("commit embeddings: %w", err)
	}

	return len(tags), nil
}

func ensureEmbeddingsTable(ctx context.Context, db *sql.DB, table string) error {
	query := fmt.Sprintf(`
CREATE TABLE IF NOT EXISTS %s (
	facet text PRIMARY KEY,
	dim int NOT NULL,
	frequency integer NOT NULL DEFAULT 0,
	weight double precision NOT NULL DEFAULT 1,
	vector double precision[] NOT NULL,
	last_trained_at timestamp without time zone NOT NULL DEFAULT now()
);
`, quoteIdentifier(table))
	if _, err := db.ExecContext(ctx, query); err != nil {
		return err
	}

	// upgrade tables created before the frequency/weight columns existed
	for _, alter := range []string{
		`ALTER TABLE %s ADD COLUMN IF NOT EXISTS frequency integer NOT NULL DEFAULT 0`,
		`ALTER TABLE %s ADD COLUMN IF NOT EXISTS weight double precision NOT NULL DEFAULT 1`,
	} {
		if _, err := db.ExecContext(ctx, fmt.Sprintf(alter, quoteIdentifier(table))); err != nil {
			return err
		}
	}
	return nil
}

// TrainingMeta records how a set of embeddings was built so the exact
// configuration of any run can be recovered later.
type TrainingMeta struct {
	Command         string    `json:"command"`
	Config          CLIConfig `json:"config"`
	NumItems        int       `json:"num_items"`
	NumTags         int       `json:"num_tags"`
	DurationSeconds float64   `json:"duration_seconds"`
	TrainedAt       time.Time `json:"trained_at"`
}

// Comment renders the metadata as a human-readable table comment.
func (m TrainingMeta) Comment() string {
	return fmt.Sprintf(
		"Generated by facetembeddings at %s\ncommand: %s\ncorpus: %d items, %d tags; took %.0fs",
		m.TrainedAt.Format(time.RFC3339),
		m.Command,
		m.NumItems,
		m.NumTags,
		m.DurationSeconds,
	)
}

// SaveTrainingComment stores the training run description as the table's
// COMMENT so it travels with pg_dump/.sql exports of the table.
func SaveTrainingComment(ctx context.Context, db *sql.DB, table string, meta TrainingMeta) error {
	commentSQL := fmt.Sprintf(
		`COMMENT ON TABLE %s IS %s`,
		quoteIdentifier(table),
		quoteLiteral(meta.Comment()),
	)
	if _, err := db.ExecContext(ctx, commentSQL); err != nil {
		return fmt.Errorf("comment on table: %w", err)
	}
	return nil
}

func quoteLiteral(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "''") + "'"
}

func quoteIdentifier(id string) string {
	if id == "" {
		return `""`
	}
	return `"` + strings.ReplaceAll(id, `"`, `""`) + `"`
}
