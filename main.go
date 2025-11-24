package main

import (
	"context"
	"flag"
	"log"
	"time"

	"gonum.org/v1/gonum/blas/blas64"
	gonumblas "gonum.org/v1/gonum/blas/gonum"
)

type CLIConfig struct {
	BatchSize int
	TableName string
	EmbeddingConfig
}

const defaultBatchSize = 20000

var defaultEmbeddingConfig = EmbeddingConfig{
	EmbeddingDim:      32,
	MinTagFrequency:   5,
	MaxTags:           20_000,
	MinCooccurrence:   1,
	MatrixType:        "ppmi",
	FactorizationType: "svd",
	ALSIterations:     15,
	ALSRegularization: 0.1,
	ALSConvergence:    1e-4,
}

func main() {
	ctx := context.Background()
	cfg := parseFlags()

	start := time.Now()

	db, err := Connect(ctx)
	if err != nil {
		log.Fatalf("connect database: %v", err)
	}
	defer db.Close()

	log.Printf("loading item tags (batch size=%d)...", cfg.BatchSize)
	items, err := LoadItemTags(ctx, db, cfg.BatchSize)
	if err != nil {
		log.Fatalf("load item tags: %v", err)
	}
	log.Printf("loaded %d items with tags", len(items))
	logItemDatasetStats(items)

	embeddings, vocab, err := RunEmbeddingPipeline(items, cfg.EmbeddingConfig)
	if err != nil {
		log.Fatalf("compute embeddings: %v", err)
	}
	log.Printf("built embeddings for %d tags", len(vocab.IndexToTag))

	log.Printf("writing embeddings to database table %q...", cfg.TableName)
	inserted, err := SaveEmbeddings(ctx, db, cfg.TableName, embeddings, cfg.EmbeddingDim)
	if err != nil {
		log.Fatalf("save embeddings: %v", err)
	}

	log.Printf("wrote %d embeddings in %s", inserted, time.Since(start).Round(time.Millisecond))
}

func logItemDatasetStats(items []ItemTags) {
	if len(items) == 0 {
		log.Printf("dataset stats: no items with tags found")
		return
	}
	var totalTags int
	maxTags := 0
	for _, item := range items {
		tagCount := len(item.Tags)
		totalTags += tagCount
		if tagCount > maxTags {
			maxTags = tagCount
		}
	}
	avgTags := float64(totalTags) / float64(len(items))
	log.Printf(
		"dataset stats: total tags=%d, avg tags/item=%.2f, max tags on single item=%d",
		totalTags,
		avgTags,
		maxTags,
	)
}

func parseFlags() CLIConfig {
	var cfg CLIConfig
	cfg.BatchSize = defaultBatchSize
	cfg.EmbeddingConfig = defaultEmbeddingConfig
	cfg.TableName = "facet_embeddings"

	flag.IntVar(&cfg.BatchSize, "batch-size", cfg.BatchSize, "Number of rows to fetch per DB batch")
	flag.IntVar(&cfg.EmbeddingDim, "embedding-dim", cfg.EmbeddingDim, "Embedding dimensionality")
	flag.IntVar(&cfg.MinTagFrequency, "min-tag-frequency", cfg.MinTagFrequency, "Minimum number of items for a tag to be included")
	flag.IntVar(&cfg.MaxTags, "max-tags", cfg.MaxTags, "Maximum number of tags to embed (0 = unlimited)")
	flag.IntVar(&cfg.MinCooccurrence, "min-cooccurrence", cfg.MinCooccurrence, "Minimum co-occurrence count to keep matrix entries")
	flag.StringVar(&cfg.TableName, "table", cfg.TableName, "Database table to write embeddings into")
	flag.StringVar(&cfg.MatrixType, "matrix-type", cfg.MatrixType, `Matrix type to use ("cooc" or "ppmi")`)
	flag.StringVar(&cfg.FactorizationType, "factorization", cfg.FactorizationType, `Factorization method ("svd" or "als")`)
	flag.IntVar(&cfg.ALSIterations, "als-iterations", cfg.ALSIterations, "Maximum ALS iterations (only used with -factorization=als)")
	flag.Float64Var(&cfg.ALSRegularization, "als-lambda", cfg.ALSRegularization, "ALS regularization parameter (only used with -factorization=als)")
	flag.Float64Var(&cfg.ALSConvergence, "als-convergence", cfg.ALSConvergence, "ALS convergence threshold for early stopping")

	flag.Parse()

	return cfg
}

func init() {
	blas64.Use(gonumblas.Implementation{})
}
