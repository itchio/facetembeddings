package main

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestBuildSparseCSR(t *testing.T) {
	// Create a small symmetric matrix with known values
	data := []float64{
		1, 2, 0,
		2, 0, 3,
		0, 3, 4,
	}
	sym := mat.NewSymDense(3, data)

	csr := BuildSparseCSR(sym)

	// Check dimensions
	r, c := csr.Dims()
	if r != 3 || c != 3 {
		t.Errorf("expected 3x3 matrix, got %dx%d", r, c)
	}

	// Check NNZ (non-zero count)
	// Non-zeros: (0,0)=1, (0,1)=2, (1,0)=2, (1,2)=3, (2,1)=3, (2,2)=4 = 6
	expectedNNZ := 6
	if csr.NNZ() != expectedNNZ {
		t.Errorf("expected %d non-zeros, got %d", expectedNNZ, csr.NNZ())
	}

	// Check specific values
	tests := []struct {
		i, j int
		want float64
	}{
		{0, 0, 1},
		{0, 1, 2},
		{0, 2, 0},
		{1, 0, 2},
		{1, 1, 0},
		{1, 2, 3},
		{2, 0, 0},
		{2, 1, 3},
		{2, 2, 4},
	}

	for _, tt := range tests {
		got := csr.At(tt.i, tt.j)
		if got != tt.want {
			t.Errorf("At(%d,%d) = %v, want %v", tt.i, tt.j, got, tt.want)
		}
	}
}

func TestBuildSparseCSR_Empty(t *testing.T) {
	// All zeros
	sym := mat.NewSymDense(3, nil)
	csr := BuildSparseCSR(sym)

	if csr.NNZ() != 0 {
		t.Errorf("expected 0 non-zeros for empty matrix, got %d", csr.NNZ())
	}
}

func TestComputeALSEmbeddings_SmallMatrix(t *testing.T) {
	// Create a simple co-occurrence matrix
	// Tags: A, B, C where A-B co-occur strongly, C is separate
	data := []float64{
		5, 4, 1,
		4, 5, 1,
		1, 1, 3,
	}
	sym := mat.NewSymDense(3, data)

	vocab := Vocabulary{
		TagToIndex: map[string]int{"A": 0, "B": 1, "C": 2},
		IndexToTag: []string{"A", "B", "C"},
		Frequency:  map[string]int{"A": 10, "B": 10, "C": 5},
	}

	cfg := ALSConfig{
		Rank:           2,
		MaxIterations:  10,
		Lambda:         0.1,
		ConvergenceEps: 1e-6,
	}

	embeddings, err := ComputeALSEmbeddings(sym, vocab, cfg)
	if err != nil {
		t.Fatalf("ComputeALSEmbeddings failed: %v", err)
	}

	// Check we got embeddings for all tags
	if len(embeddings) != 3 {
		t.Errorf("expected 3 embeddings, got %d", len(embeddings))
	}

	for _, tag := range []string{"A", "B", "C"} {
		emb, ok := embeddings[tag]
		if !ok {
			t.Errorf("missing embedding for tag %q", tag)
			continue
		}
		if len(emb) != 2 {
			t.Errorf("embedding for %q has dim %d, want 2", tag, len(emb))
		}
	}

	// A and B should be more similar to each other than to C
	simAB := cosineSimilarity(embeddings["A"], embeddings["B"])
	simAC := cosineSimilarity(embeddings["A"], embeddings["C"])
	simBC := cosineSimilarity(embeddings["B"], embeddings["C"])

	if simAB <= simAC || simAB <= simBC {
		t.Errorf("expected A-B similarity (%.3f) > A-C (%.3f) and B-C (%.3f)", simAB, simAC, simBC)
	}
}

func TestComputeALSEmbeddings_InvalidRank(t *testing.T) {
	sym := mat.NewSymDense(3, nil)
	vocab := Vocabulary{
		TagToIndex: map[string]int{"A": 0, "B": 1, "C": 2},
		IndexToTag: []string{"A", "B", "C"},
	}

	// Rank 0 should fail
	_, err := ComputeALSEmbeddings(sym, vocab, ALSConfig{Rank: 0, MaxIterations: 5, Lambda: 0.1})
	if err == nil {
		t.Error("expected error for rank 0")
	}

	// Rank > n should fail
	_, err = ComputeALSEmbeddings(sym, vocab, ALSConfig{Rank: 10, MaxIterations: 5, Lambda: 0.1})
	if err == nil {
		t.Error("expected error for rank > matrix dimension")
	}
}

func TestComputeALSEmbeddings_Convergence(t *testing.T) {
	// Create a low-rank matrix that ALS should be able to approximate well
	// V = W * W^T where W is 5x2
	n, k := 5, 2
	W := mat.NewDense(n, k, []float64{
		1, 0,
		0.9, 0.1,
		0.1, 0.9,
		0, 1,
		0.5, 0.5,
	})

	// Compute V = W * W^T
	var V mat.Dense
	V.Mul(W, W.T())

	// Convert to SymDense
	sym := mat.NewSymDense(n, nil)
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			sym.SetSym(i, j, V.At(i, j))
		}
	}

	vocab := Vocabulary{
		TagToIndex: map[string]int{"0": 0, "1": 1, "2": 2, "3": 3, "4": 4},
		IndexToTag: []string{"0", "1", "2", "3", "4"},
	}

	cfg := ALSConfig{
		Rank:           k,
		MaxIterations:  50,
		Lambda:         0.01,
		ConvergenceEps: 1e-8,
	}

	embeddings, err := ComputeALSEmbeddings(sym, vocab, cfg)
	if err != nil {
		t.Fatalf("ComputeALSEmbeddings failed: %v", err)
	}

	// Reconstruct and check error is small
	var totalError float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			original := sym.At(i, j)
			reconstructed := dotProduct(embeddings[vocab.IndexToTag[i]], embeddings[vocab.IndexToTag[j]])
			diff := original - reconstructed
			totalError += diff * diff
		}
	}
	rmse := math.Sqrt(totalError / float64(n*n))

	// RMSE should be reasonably small for a low-rank matrix
	if rmse > 0.5 {
		t.Errorf("reconstruction RMSE = %.4f, expected < 0.5", rmse)
	}
}

func TestInitializeRandomMatrix(t *testing.T) {
	m := mat.NewDense(10, 5, nil)
	initializeRandomMatrix(m, 42)

	// Check that values are non-zero and small
	r, c := m.Dims()
	var sum float64
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			v := m.At(i, j)
			sum += v * v
		}
	}

	// Should have some variance
	if sum == 0 {
		t.Error("matrix should have non-zero values after initialization")
	}

	// Values should be small (scale is 0.01)
	avgMagnitude := math.Sqrt(sum / float64(r*c))
	if avgMagnitude > 0.1 {
		t.Errorf("average magnitude = %.4f, expected < 0.1", avgMagnitude)
	}
}

func TestInitializeRandomMatrix_Deterministic(t *testing.T) {
	m1 := mat.NewDense(5, 3, nil)
	m2 := mat.NewDense(5, 3, nil)

	initializeRandomMatrix(m1, 123)
	initializeRandomMatrix(m2, 123)

	// Same seed should produce same values
	r, c := m1.Dims()
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			if m1.At(i, j) != m2.At(i, j) {
				t.Errorf("same seed should produce same values at (%d,%d)", i, j)
			}
		}
	}
}

func TestExtractEmbeddingsFromW(t *testing.T) {
	W := mat.NewDense(3, 2, []float64{
		1, 2,
		3, 4,
		5, 6,
	})

	vocab := Vocabulary{
		TagToIndex: map[string]int{"X": 0, "Y": 1, "Z": 2},
		IndexToTag: []string{"X", "Y", "Z"},
	}

	embeddings := extractEmbeddingsFromW(W, vocab)

	expected := map[string][]float64{
		"X": {1, 2},
		"Y": {3, 4},
		"Z": {5, 6},
	}

	for tag, want := range expected {
		got, ok := embeddings[tag]
		if !ok {
			t.Errorf("missing embedding for %q", tag)
			continue
		}
		if len(got) != len(want) {
			t.Errorf("embedding for %q has len %d, want %d", tag, len(got), len(want))
			continue
		}
		for i := range want {
			if got[i] != want[i] {
				t.Errorf("embedding[%q][%d] = %v, want %v", tag, i, got[i], want[i])
			}
		}
	}
}

// Helper functions

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func dotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0
	}
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
