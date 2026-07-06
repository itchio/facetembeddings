package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// WriteEmbeddingsCSV writes embeddings to a CSV file compatible with the facet_embeddings table.
// Columns: facet, dim, frequency, weight, vector, last_trained_at. The vector is formatted as a
// Postgres array literal. freq holds the per-facet document frequency (number of games the facet
// appeared on at training time); weights holds the SIF pooling weight already baked into each
// vector's magnitude.
func WriteEmbeddingsCSV(path string, embeddings map[string][]float64, freq map[string]int, weights map[string]float64, dim int, ts time.Time) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	if err := w.Write([]string{"facet", "dim", "frequency", "weight", "vector", "last_trained_at"}); err != nil {
		return fmt.Errorf("write header: %w", err)
	}

	tags := make([]string, 0, len(embeddings))
	for tag := range embeddings {
		tags = append(tags, tag)
	}
	sort.Strings(tags)

	timestamp := ts.Format(time.RFC3339)

	for _, tag := range tags {
		vec := embeddings[tag]
		vecStr := formatVectorForPostgres(vec)
		weight := 1.0
		if w, ok := weights[tag]; ok {
			weight = w
		}
		row := []string{
			tag,
			strconv.Itoa(dim),
			strconv.Itoa(freq[tag]),
			strconv.FormatFloat(weight, 'g', -1, 64),
			vecStr,
			timestamp,
		}
		if err := w.Write(row); err != nil {
			return fmt.Errorf("write row for %s: %w", tag, err)
		}
	}

	w.Flush()
	if err := w.Error(); err != nil {
		return fmt.Errorf("flush csv: %w", err)
	}

	return nil
}

func formatVectorForPostgres(vec []float64) string {
	parts := make([]string, len(vec))
	for i, v := range vec {
		parts[i] = strconv.FormatFloat(v, 'g', -1, 64)
	}
	return "{" + strings.Join(parts, ",") + "}"
}
