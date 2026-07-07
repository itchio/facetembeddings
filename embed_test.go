package main

import (
	"database/sql"
	"math"
	"testing"
)

func testVocab(tags ...string) Vocabulary {
	v := Vocabulary{
		TagToIndex: map[string]int{},
		IndexToTag: tags,
		Frequency:  map[string]int{},
	}
	for i, t := range tags {
		v.TagToIndex[t] = i
	}
	return v
}

func TestItemIncrement(t *testing.T) {
	item := ItemTags{Weight: 0.5}

	if got := itemIncrement(item, 5, false); got != 0.5 {
		t.Errorf("no norm: want 0.5, got %g", got)
	}
	if got := itemIncrement(item, 5, true); got != 0.125 {
		t.Errorf("per-game norm: want 0.5/4, got %g", got)
	}
	if got := itemIncrement(item, 1, true); got != 0.5 {
		t.Errorf("single-tag item should not divide by zero: got %g", got)
	}

	// zero-value weight (eg. hand-built test items) falls back to 1
	if got := itemIncrement(ItemTags{}, 3, false); got != 1 {
		t.Errorf("unset weight should default to 1, got %g", got)
	}
}

func TestItemQualityWeight(t *testing.T) {
	// unrated games get the midpoint of [0.25, 1]
	if got := itemQualityWeight(sql.NullFloat64{}); got != 0.625 {
		t.Errorf("null rating: want 0.625, got %g", got)
	}
	if got := itemQualityWeight(sql.NullFloat64{Valid: true, Float64: 0}); got != 0.625 {
		t.Errorf("zero rating: want 0.625, got %g", got)
	}

	if got := itemQualityWeight(sql.NullFloat64{Valid: true, Float64: 5}); got != 1 {
		t.Errorf("top rating: want 1, got %g", got)
	}
	low := itemQualityWeight(sql.NullFloat64{Valid: true, Float64: 1})
	if low != 0.4 {
		t.Errorf("1-star rating: want 0.4, got %g", low)
	}

	// ratings above scale are clamped
	if got := itemQualityWeight(sql.NullFloat64{Valid: true, Float64: 99}); got != 1 {
		t.Errorf("clamped rating: want 1, got %g", got)
	}
}

func TestBuildPPMIMatrixAlphaOneMatchesClassic(t *testing.T) {
	vocab := testVocab("a", "b", "c")
	items := []ItemTags{
		{Tags: []string{"a", "b"}, Weight: 1},
		{Tags: []string{"a", "b"}, Weight: 1},
		{Tags: []string{"a", "c"}, Weight: 1},
		{Tags: []string{"b", "c"}, Weight: 1},
	}

	cfg := EmbeddingConfig{PPMIAlpha: 1}
	ppmi := BuildPPMIMatrix(items, vocab, cfg)

	// counts: ab=2, ac=1, bc=1, totalPairs=4
	// marginals over 2*totalPairs=8: p_a=3/8, p_b=3/8, p_c=2/8
	// pmi(a,b) = log2((2/4) / (3/8 * 3/8)) = log2(32/9)
	want := math.Log2(32.0 / 9.0)
	if got := ppmi.At(0, 1); math.Abs(got-want) > 1e-12 {
		t.Errorf("pmi(a,b): want %g, got %g", want, got)
	}

	// pmi(a,c) = log2((1/4) / (3/8 * 2/8)) = log2(8/3)
	want = math.Log2(8.0 / 3.0)
	if got := ppmi.At(0, 2); math.Abs(got-want) > 1e-12 {
		t.Errorf("pmi(a,c): want %g, got %g", want, got)
	}
}

func TestBuildPPMIMatrixSmoothingDampsRareTags(t *testing.T) {
	vocab := testVocab("common1", "common2", "rare1", "rare2")

	// two common tags co-occur a lot; two rare tags co-occur once
	var items []ItemTags
	for i := 0; i < 50; i++ {
		items = append(items, ItemTags{Tags: []string{"common1", "common2"}, Weight: 1})
	}
	items = append(items, ItemTags{Tags: []string{"rare1", "rare2"}, Weight: 1})

	classic := BuildPPMIMatrix(items, vocab, EmbeddingConfig{PPMIAlpha: 1})
	smoothed := BuildPPMIMatrix(items, vocab, EmbeddingConfig{PPMIAlpha: 0.75})

	rareClassic := classic.At(2, 3)
	rareSmoothed := smoothed.At(2, 3)
	commonClassic := classic.At(0, 1)
	commonSmoothed := smoothed.At(0, 1)

	// smoothing should reduce the rare pair's advantage over the common pair
	classicGap := rareClassic - commonClassic
	smoothedGap := rareSmoothed - commonSmoothed
	if smoothedGap >= classicGap {
		t.Errorf("smoothing should shrink rare-pair PMI advantage: classic gap %g, smoothed gap %g",
			classicGap, smoothedGap)
	}
}

func TestCooccurrenceQualityWeightingAndNorm(t *testing.T) {
	vocab := testVocab("a", "b", "c")

	items := []ItemTags{
		{Tags: []string{"a", "b"}, Weight: 0.25},
		{Tags: []string{"a", "c"}, Weight: 1},
	}

	co := BuildCooccurrenceMatrix(items, vocab, EmbeddingConfig{})

	if got := co.At(0, 1); got != 0.25 {
		t.Errorf("low-quality pair should count 0.25, got %g", got)
	}
	if got := co.At(0, 2); got != 1 {
		t.Errorf("high-quality pair should count 1, got %g", got)
	}

	// per-game norm divides a 3-tag game's increments by 2
	normed := BuildCooccurrenceMatrix([]ItemTags{
		{Tags: []string{"a", "b", "c"}, Weight: 1},
	}, vocab, EmbeddingConfig{PerGameNorm: true})

	if got := normed.At(0, 1); got != 0.5 {
		t.Errorf("per-game norm: 3-tag game pair should count 1/2, got %g", got)
	}
}
