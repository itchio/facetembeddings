package main

import (
	"context"
	"flag"
	"log"
	"os"
	"path/filepath"
	"time"

	"gonum.org/v1/gonum/blas/blas64"
	gonumblas "gonum.org/v1/gonum/blas/gonum"
)

type CLIConfig struct {
	BatchSize  int
	TableName  string
	OutputFile string
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
	if len(os.Args) < 2 {
		printUsageAndExit()
	}

	subcmd := os.Args[1]
	switch subcmd {
	case "generate":
		if err := runGenerateCLI(os.Args[2:]); err != nil {
			log.Fatalf("generate: %v", err)
		}
	case "cluster":
		if err := runClusterCLI(os.Args[2:]); err != nil {
			log.Fatalf("cluster: %v", err)
		}
	case "-h", "--help", "help":
		printUsageAndExit()
	default:
		log.Printf("unknown subcommand: %q", subcmd)
		printUsageAndExit()
	}
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

func parseGenerateFlags(args []string) CLIConfig {
	var cfg CLIConfig
	cfg.BatchSize = defaultBatchSize
	cfg.EmbeddingConfig = defaultEmbeddingConfig
	cfg.TableName = "facet_embeddings"
	cfg.OutputFile = ""

	fs := flag.NewFlagSet("generate", flag.ExitOnError)

	fs.IntVar(&cfg.BatchSize, "batch-size", cfg.BatchSize, "Number of rows to fetch per DB batch")
	fs.IntVar(&cfg.EmbeddingDim, "embedding-dim", cfg.EmbeddingDim, "Embedding dimensionality")
	fs.IntVar(&cfg.MinTagFrequency, "min-tag-frequency", cfg.MinTagFrequency, "Minimum number of items for a tag to be included")
	fs.IntVar(&cfg.MaxTags, "max-tags", cfg.MaxTags, "Maximum number of tags to embed (0 = unlimited)")
	fs.IntVar(&cfg.MinCooccurrence, "min-cooccurrence", cfg.MinCooccurrence, "Minimum co-occurrence count to keep matrix entries")
	fs.StringVar(&cfg.TableName, "table", cfg.TableName, "Database table to write embeddings into")
	fs.StringVar(&cfg.OutputFile, "output-file", cfg.OutputFile, "If set, write embeddings to CSV file instead of the database")
	fs.StringVar(&cfg.MatrixType, "matrix-type", cfg.MatrixType, `Matrix type to use ("cooc" or "ppmi")`)
	fs.StringVar(&cfg.FactorizationType, "factorization", cfg.FactorizationType, `Factorization method ("svd" or "als")`)
	fs.IntVar(&cfg.ALSIterations, "als-iterations", cfg.ALSIterations, "Maximum ALS iterations (only used with -factorization=als)")
	fs.Float64Var(&cfg.ALSRegularization, "als-lambda", cfg.ALSRegularization, "ALS regularization parameter (only used with -factorization=als)")
	fs.Float64Var(&cfg.ALSConvergence, "als-convergence", cfg.ALSConvergence, "ALS convergence threshold for early stopping")

	if err := fs.Parse(args); err != nil {
		log.Fatalf("parse flags: %v", err)
	}

	return cfg
}

func init() {
	blas64.Use(gonumblas.Implementation{})
}

func runGenerateCLI(args []string) error {
	ctx := context.Background()
	cfg := parseGenerateFlags(args)

	start := time.Now()

	db, err := Connect(ctx)
	if err != nil {
		return err
	}
	defer db.Close()

	log.Printf("loading item tags (batch size=%d)...", cfg.BatchSize)
	items, err := LoadItemTags(ctx, db, cfg.BatchSize)
	if err != nil {
		return err
	}
	log.Printf("loaded %d items with tags", len(items))
	logItemDatasetStats(items)

	embeddings, vocab, err := RunEmbeddingPipeline(items, cfg.EmbeddingConfig)
	if err != nil {
		return err
	}
	log.Printf("built embeddings for %d tags", len(vocab.IndexToTag))

	if cfg.OutputFile != "" {
		log.Printf("writing embeddings to file %q...", cfg.OutputFile)
		if err := WriteEmbeddingsCSV(cfg.OutputFile, embeddings, cfg.EmbeddingDim, time.Now()); err != nil {
			return err
		}
		log.Printf("wrote %d embeddings to %q in %s", len(embeddings), cfg.OutputFile, time.Since(start).Round(time.Millisecond))
		return nil
	}

	log.Printf("writing embeddings to database table %q...", cfg.TableName)
	inserted, err := SaveEmbeddings(ctx, db, cfg.TableName, embeddings, cfg.EmbeddingDim)
	if err != nil {
		return err
	}

	log.Printf("wrote %d embeddings in %s", inserted, time.Since(start).Round(time.Millisecond))
	return nil
}

// runClusterCLI handles the "cluster" subcommand.
func runClusterCLI(args []string) error {
	ctx := context.Background()

	var cfg ClusterConfig
	cfg.ClusterColumn = "vector"
	cfg.NumClusters = 20
	cfg.MaxIterations = 50
	cfg.Tolerance = 1e-4
	cfg.ExamplesPerRow = 5

	fs := flag.NewFlagSet("cluster", flag.ExitOnError)
	fs.StringVar(&cfg.ClusterTable, "cluster-table", "", "Table containing vectors to cluster (must have columns id, vector column)")
	fs.StringVar(&cfg.ClusterColumn, "cluster-column", cfg.ClusterColumn, "Vector column name to use for clustering")
	fs.IntVar(&cfg.NumClusters, "clusters", cfg.NumClusters, "Number of clusters (k)")
	fs.IntVar(&cfg.MaxIterations, "max-iterations", cfg.MaxIterations, "Maximum k-means iterations")
	fs.Float64Var(&cfg.Tolerance, "tolerance", cfg.Tolerance, "Centroid movement tolerance for convergence (1 - cosine)")
	fs.Int64Var(&cfg.Seed, "seed", cfg.Seed, "Random seed (0 = random)")
	fs.IntVar(&cfg.ExamplesPerRow, "examples", cfg.ExamplesPerRow, "Number of example IDs to print per cluster")

	if err := fs.Parse(args); err != nil {
		return err
	}

	db, err := Connect(ctx)
	if err != nil {
		return err
	}
	defer db.Close()

	return RunClusterCommand(ctx, db, cfg)
}

func printUsageAndExit() {
	exe := filepath.Base(os.Args[0])
	log.Printf("usage: %s <subcommand> [options]", exe)
	log.Printf("subcommands:")
	log.Printf("  generate   Build facet embeddings and write to DB or CSV")
	log.Printf("  cluster    Cluster rows in a table containing id + vector")
	os.Exit(2)
}
