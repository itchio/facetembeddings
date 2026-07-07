package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"strings"
	"time"

	"gonum.org/v1/gonum/blas/blas64"
	gonumblas "gonum.org/v1/gonum/blas/gonum"
)

type CLIConfig struct {
	BatchSize  int
	TableName  string
	OutputFile string

	// QualityWeight scales each game's co-occurrence contribution by its
	// weighted_rating (see itemQualityWeight).
	QualityWeight bool

	EmbeddingConfig
}

const defaultBatchSize = 20000

var defaultEmbeddingConfig = EmbeddingConfig{
	EmbeddingDim:      256,
	MinTagFrequency:   5,
	MaxTags:           20_000,
	MinCooccurrence:   1,
	MatrixType:        "ppmi",
	FactorizationType: "svd",
	SIFParam:          0.001,
	PPMIAlpha:         0.75,
	PerGameNorm:       true,
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

	log.Printf("effective config: %+v", cfg)

	log.Printf("loading item tags (batch size=%d, quality weight=%v)...", cfg.BatchSize, cfg.QualityWeight)
	items, err := LoadItemTags(ctx, db, cfg.BatchSize, cfg.QualityWeight)
	if err != nil {
		log.Fatalf("load item tags: %v", err)
	}
	log.Printf("loaded %d items with tags", len(items))
	logItemDatasetStats(items)

	embeddings, weights, vocab, err := RunEmbeddingPipeline(items, cfg.EmbeddingConfig)
	if err != nil {
		log.Fatalf("compute embeddings: %v", err)
	}
	log.Printf("built embeddings for %d tags", len(vocab.IndexToTag))

	meta := TrainingMeta{
		Command:         cfg.CommandLine(),
		Config:          cfg,
		NumItems:        len(items),
		NumTags:         len(vocab.IndexToTag),
		DurationSeconds: time.Since(start).Seconds(),
		TrainedAt:       time.Now(),
	}

	if cfg.OutputFile != "" {
		log.Printf("writing embeddings to file %q...", cfg.OutputFile)
		if err := WriteEmbeddingsCSV(cfg.OutputFile, embeddings, vocab.Frequency, weights, cfg.EmbeddingDim, time.Now()); err != nil {
			log.Fatalf("write embeddings to file: %v", err)
		}
		if err := WriteMetaJSON(cfg.OutputFile, meta); err != nil {
			log.Fatalf("write training meta: %v", err)
		}
		log.Printf("wrote %d embeddings to %q (meta: %s.meta.json) in %s", len(embeddings), cfg.OutputFile, cfg.OutputFile, time.Since(start).Round(time.Millisecond))
		return
	}

	log.Printf("writing embeddings to database table %q...", cfg.TableName)
	inserted, err := SaveEmbeddings(ctx, db, cfg.TableName, embeddings, vocab.Frequency, weights, cfg.EmbeddingDim)
	if err != nil {
		log.Fatalf("save embeddings: %v", err)
	}

	if err := SaveTrainingComment(ctx, db, cfg.TableName, meta); err != nil {
		log.Fatalf("save training comment: %v", err)
	}

	log.Printf("wrote %d embeddings (run recorded in table comment) in %s", inserted, time.Since(start).Round(time.Millisecond))
}

// CommandLine reconstructs the equivalent CLI invocation for this config so
// it can be recorded alongside the generated embeddings.
func (cfg CLIConfig) CommandLine() string {
	args := []string{
		"facetembeddings",
		fmt.Sprintf("-table=%s", cfg.TableName),
		fmt.Sprintf("-embedding-dim=%d", cfg.EmbeddingDim),
		fmt.Sprintf("-min-tag-frequency=%d", cfg.MinTagFrequency),
		fmt.Sprintf("-max-tags=%d", cfg.MaxTags),
		fmt.Sprintf("-min-cooccurrence=%d", cfg.MinCooccurrence),
		fmt.Sprintf("-matrix-type=%s", cfg.MatrixType),
		fmt.Sprintf("-factorization=%s", cfg.FactorizationType),
		fmt.Sprintf("-sif-a=%g", cfg.SIFParam),
		fmt.Sprintf("-ppmi-alpha=%g", cfg.PPMIAlpha),
		fmt.Sprintf("-per-game-norm=%v", cfg.PerGameNorm),
		fmt.Sprintf("-quality-weight=%v", cfg.QualityWeight),
		fmt.Sprintf("-batch-size=%d", cfg.BatchSize),
	}

	if cfg.FactorizationType == "als" {
		args = append(args,
			fmt.Sprintf("-als-iterations=%d", cfg.ALSIterations),
			fmt.Sprintf("-als-lambda=%g", cfg.ALSRegularization),
			fmt.Sprintf("-als-convergence=%g", cfg.ALSConvergence),
		)
	}

	if cfg.OutputFile != "" {
		args = append(args, fmt.Sprintf("-output-file=%s", cfg.OutputFile))
	}

	return strings.Join(args, " ")
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
	cfg.OutputFile = ""

	flag.IntVar(&cfg.BatchSize, "batch-size", cfg.BatchSize, "Number of rows to fetch per DB batch")
	flag.IntVar(&cfg.EmbeddingDim, "embedding-dim", cfg.EmbeddingDim, "Embedding dimensionality")
	flag.IntVar(&cfg.MinTagFrequency, "min-tag-frequency", cfg.MinTagFrequency, "Minimum number of items for a tag to be included")
	flag.IntVar(&cfg.MaxTags, "max-tags", cfg.MaxTags, "Maximum number of tags to embed (0 = unlimited)")
	flag.IntVar(&cfg.MinCooccurrence, "min-cooccurrence", cfg.MinCooccurrence, "Minimum co-occurrence count to keep matrix entries")
	flag.StringVar(&cfg.TableName, "table", cfg.TableName, "Database table to write embeddings into")
	flag.StringVar(&cfg.OutputFile, "output-file", cfg.OutputFile, "If set, write embeddings to CSV file instead of the database")
	flag.StringVar(&cfg.MatrixType, "matrix-type", cfg.MatrixType, `Matrix type to use ("cooc" or "ppmi")`)
	flag.Float64Var(&cfg.SIFParam, "sif-a", cfg.SIFParam, "SIF pooling-weight parameter a in a/(a+p); vectors are scaled by their weight (<= 0 disables, vectors stay unit length)")
	flag.Float64Var(&cfg.PPMIAlpha, "ppmi-alpha", cfg.PPMIAlpha, "PPMI context-distribution smoothing exponent (1 = classic PPMI)")
	flag.BoolVar(&cfg.PerGameNorm, "per-game-norm", cfg.PerGameNorm, "Normalize each game's co-occurrence mass by its tag count (prevents heavily-tagged pages from dominating)")
	flag.BoolVar(&cfg.QualityWeight, "quality-weight", true, "Weight each game's co-occurrence contribution by its weighted_rating")
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
