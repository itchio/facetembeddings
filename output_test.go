package main

import (
	"encoding/csv"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

func TestWriteEmbeddingsCSV(t *testing.T) {
	path := filepath.Join(t.TempDir(), "embeddings.csv")

	embeddings := map[string][]float64{
		"tg.horror": {0.5, -0.25},
		"tg.puzzle": {1, 0},
	}
	freq := map[string]int{
		"tg.horror": 120,
		"tg.puzzle": 45,
	}
	weights := map[string]float64{
		"tg.horror": 0.5,
		"tg.puzzle": 0.75,
	}
	ts := time.Date(2026, 7, 6, 12, 0, 0, 0, time.UTC)

	if err := WriteEmbeddingsCSV(path, embeddings, freq, weights, 2, ts); err != nil {
		t.Fatalf("WriteEmbeddingsCSV: %v", err)
	}

	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open output: %v", err)
	}
	defer f.Close()

	rows, err := csv.NewReader(f).ReadAll()
	if err != nil {
		t.Fatalf("read csv: %v", err)
	}

	expected := [][]string{
		{"facet", "dim", "frequency", "weight", "vector", "last_trained_at"},
		{"tg.horror", "2", "120", "0.5", "{0.5,-0.25}", "2026-07-06T12:00:00Z"},
		{"tg.puzzle", "2", "45", "0.75", "{1,0}", "2026-07-06T12:00:00Z"},
	}

	if !reflect.DeepEqual(rows, expected) {
		t.Errorf("csv output mismatch\ngot:  %v\nwant: %v", rows, expected)
	}
}

func TestComputeFacetWeights(t *testing.T) {
	vocab := Vocabulary{
		Frequency: map[string]int{
			"m.free":    800, // stopword-common
			"tg.action": 100,
			"tg.rare":   1,
		},
	}

	weights := ComputeFacetWeights(vocab, 1000, 0.001)

	// p=0.8 gives sif ~0.00125: ultra-common facets are crushed
	if weights["m.free"] >= 0.01 {
		t.Errorf("m.free should be crushed by SIF, got %g", weights["m.free"])
	}

	// rarer tags weigh more
	if weights["tg.rare"] <= weights["tg.action"] {
		t.Errorf("rare tag should outweigh common tag: rare=%g action=%g",
			weights["tg.rare"], weights["tg.action"])
	}
	if weights["tg.rare"] <= 0.4 || weights["tg.rare"] > 1 {
		t.Errorf("tg.rare weight out of expected range: %g", weights["tg.rare"])
	}

	// a <= 0 disables weighting entirely, including floors
	disabled := ComputeFacetWeights(vocab, 1000, 0)
	for tag, w := range disabled {
		if w != 1 {
			t.Errorf("disabled weighting should give weight 1 for %s, got %g", tag, w)
		}
	}
}

func TestApplyFacetWeights(t *testing.T) {
	embeddings := map[string][]float64{
		"tg.a": {1, 0},
		"tg.b": {0, 1},
	}

	ApplyFacetWeights(embeddings, map[string]float64{
		"tg.a": 0.5,
		// tg.b intentionally missing: should stay untouched
	})

	if !reflect.DeepEqual(embeddings["tg.a"], []float64{0.5, 0}) {
		t.Errorf("tg.a not scaled: %v", embeddings["tg.a"])
	}
	if !reflect.DeepEqual(embeddings["tg.b"], []float64{0, 1}) {
		t.Errorf("tg.b should be unchanged: %v", embeddings["tg.b"])
	}
}

func TestCommandLineReconstruction(t *testing.T) {
	cfg := CLIConfig{
		BatchSize:       20000,
		TableName:       "facet_embeddings",
		QualityWeight:   true,
		EmbeddingConfig: defaultEmbeddingConfig,
	}

	cmd := cfg.CommandLine()

	for _, want := range []string{
		"facetembeddings ",
		"-embedding-dim=256",
		"-sif-a=0.001",
		"-ppmi-alpha=0.75",
		"-per-game-norm=true",
		"-quality-weight=true",
		"-matrix-type=ppmi",
	} {
		if !strings.Contains(cmd, want) {
			t.Errorf("command missing %q: %s", want, cmd)
		}
	}

	// ALS flags only appear for ALS runs
	if strings.Contains(cmd, "-als-") {
		t.Errorf("svd run should not include ALS flags: %s", cmd)
	}
}

func TestQuoteLiteral(t *testing.T) {
	if got := quoteLiteral("it's a 'test'"); got != "'it''s a ''test'''" {
		t.Errorf("bad escaping: %s", got)
	}
}
