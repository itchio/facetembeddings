package main

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// synthetic corpus with cluster structure: two "genres" whose tags co-occur
// within but rarely across, plus noise
func syntheticItems(rng *rand.Rand, nItems int) ([]ItemTags, Vocabulary) {
	var tags []string
	for i := 0; i < 30; i++ {
		tags = append(tags, fmt.Sprintf("tg.a%d", i), fmt.Sprintf("tg.b%d", i))
	}

	items := make([]ItemTags, nItems)
	for i := range items {
		cluster := "a"
		if rng.Intn(2) == 1 {
			cluster = "b"
		}
		n := 3 + rng.Intn(5)
		seen := map[string]bool{}
		var t []string
		for len(t) < n {
			tag := fmt.Sprintf("tg.%s%d", cluster, rng.Intn(30))
			if rng.Float64() < 0.1 { // cross-cluster noise
				other := "b"
				if cluster == "b" {
					other = "a"
				}
				tag = fmt.Sprintf("tg.%s%d", other, rng.Intn(30))
			}
			if !seen[tag] {
				seen[tag] = true
				t = append(t, tag)
			}
		}
		items[i] = ItemTags{GameID: int64(i), Tags: t, Weight: 1}
	}

	vocab := Vocabulary{
		TagToIndex: map[string]int{},
		Frequency:  map[string]int{},
	}
	for _, it := range items {
		for _, tag := range it.Tags {
			if _, ok := vocab.TagToIndex[tag]; !ok {
				vocab.TagToIndex[tag] = len(vocab.IndexToTag)
				vocab.IndexToTag = append(vocab.IndexToTag, tag)
			}
			vocab.Frequency[tag]++
		}
	}
	return items, vocab
}

func TestSparsePPMIMatchesDense(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	items, vocab := syntheticItems(rng, 2000)
	cfg := EmbeddingConfig{MatrixType: "ppmi", PPMIAlpha: 0.75, PerGameNorm: true, MinCooccurrence: 2}

	dense := BuildPPMIMatrix(items, vocab, cfg)

	sm, err := BuildSparseMatrix(items, vocab, cfg)
	if err != nil {
		t.Fatalf("BuildSparseMatrix: %v", err)
	}

	n := len(vocab.IndexToTag)
	var checked int
	for i := 0; i < n; i++ {
		row := map[int]float64{}
		for p := sm.rowP[i]; p < sm.rowP[i+1]; p++ {
			row[int(sm.cols[p])] = sm.vals[p]
		}
		for j := 0; j < n; j++ {
			want := dense.At(i, j)
			got := row[j]
			if math.Abs(want-got) > 1e-9 {
				t.Fatalf("ppmi mismatch at (%d,%d): dense=%g sparse=%g", i, j, want, got)
			}
			if want != 0 {
				checked++
			}
		}
	}
	if checked == 0 {
		t.Fatal("no non-zero entries compared")
	}
}

func TestRSVDMatchesExactSVDNeighbors(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	items, vocab := syntheticItems(rng, 3000)
	cfg := EmbeddingConfig{MatrixType: "ppmi", PPMIAlpha: 0.75, PerGameNorm: true}
	dim := 16

	dense := BuildPPMIMatrix(items, vocab, cfg)
	exact, err := ComputeEmbeddings(dense, vocab, dim)
	if err != nil {
		t.Fatalf("exact svd: %v", err)
	}

	sm, err := BuildSparseMatrix(items, vocab, cfg)
	if err != nil {
		t.Fatalf("BuildSparseMatrix: %v", err)
	}
	approx, err := ComputeRSVDEmbeddings(sm, vocab, dim)
	if err != nil {
		t.Fatalf("rsvd: %v", err)
	}

	// compare top-5 neighbor overlap for every tag; the subspaces are only
	// defined up to rotation, so compare geometry (neighbor sets), not raw
	// coordinates
	var totalOverlap, count float64
	for _, tag := range vocab.IndexToTag {
		e := topNeighbors(exact, tag, 5)
		a := topNeighbors(approx, tag, 5)
		set := map[string]bool{}
		for _, x := range e {
			set[x] = true
		}
		overlap := 0
		for _, x := range a {
			if set[x] {
				overlap++
			}
		}
		totalOverlap += float64(overlap) / 5
		count++
	}
	avg := totalOverlap / count
	if avg < 0.8 {
		t.Errorf("rsvd neighbor overlap with exact svd too low: %.3f (want >= 0.8)", avg)
	}
}

func topNeighbors(embeddings map[string][]float64, facet string, k int) []string {
	type scored struct {
		f string
		s float64
	}
	var all []scored
	q := embeddings[facet]
	for other, v := range embeddings {
		if other == facet {
			continue
		}
		all = append(all, scored{other, cosine(q, v)})
	}
	// selection of top k
	out := make([]string, 0, k)
	for len(out) < k && len(all) > 0 {
		best := 0
		for i := range all {
			if all[i].s > all[best].s {
				best = i
			}
		}
		out = append(out, all[best].f)
		all[best] = all[len(all)-1]
		all = all[:len(all)-1]
	}
	return out
}
