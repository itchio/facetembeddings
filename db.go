package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"os"
	"sort"
	"strings"

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
}

const selectTagsQuery = `
SELECT game_id, tsvector_to_array(facets) AS tags
FROM games_search
WHERE facets IS NOT NULL AND game_id > $1
ORDER BY game_id
LIMIT $2
`

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

// streams game/tag rows ordered by game_id in batches
func LoadItemTags(ctx context.Context, db *sql.DB, batchSize int) ([]ItemTags, error) {
	if batchSize <= 0 {
		batchSize = 10_000
	}

	var items []ItemTags
	lastID := int64(-1)

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
			if err := rows.Scan(&item.GameID, &tagArray); err != nil {
				rows.Close()
				return nil, fmt.Errorf("scan item tags: %w", err)
			}
			item.Tags = append(item.Tags[:0], tagArray...)

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

	return items, nil
}

// truncates and repopulates the facet embeddings table with the resulting vectors
func SaveEmbeddings(ctx context.Context, db *sql.DB, table string, embeddings map[string][]float64, dim int) (int, error) {
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
INSERT INTO %s (facet, dim, vector, last_trained_at)
VALUES ($1, $2, $3, now())
`, quoteIdentifier(table))

	tags := make([]string, 0, len(embeddings))
	for tag := range embeddings {
		tags = append(tags, tag)
	}
	sort.Strings(tags)

	for _, tag := range tags {
		vec := embeddings[tag]
		if _, err := tx.ExecContext(ctx, insertSQL, tag, dim, pq.Array(vec)); err != nil {
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
	vector double precision[] NOT NULL,
	last_trained_at timestamp without time zone NOT NULL DEFAULT now()
);
`, quoteIdentifier(table))
	_, err := db.ExecContext(ctx, query)
	return err
}

func quoteIdentifier(id string) string {
	if id == "" {
		return `""`
	}
	return `"` + strings.ReplaceAll(id, `"`, `""`) + `"`
}
