package main

import (
	"math"
	"testing"
)

func TestSphericalKMeansSeparatesOrthogonalClusters(t *testing.T) {
	// Build two clear clusters around the unit axes.
	items := []ClusterItem{
		{ID: 1, Vector: []float64{1, 0}},
		{ID: 2, Vector: []float64{0.9, 0.1}},
		{ID: 3, Vector: []float64{0, 1}},
		{ID: 4, Vector: []float64{0.1, 0.9}},
	}
	items = normalizeClusterItems(items)

	assignments, centroids, iter, shift, err := SphericalKMeans(items, 2, 20, 1e-6, 42)
	if err != nil {
		t.Fatalf("spherical k-means failed: %v", err)
	}
	if iter <= 0 || iter > 20 {
		t.Fatalf("unexpected iteration count: %d", iter)
	}
	if shift > 0.1 {
		t.Fatalf("expected centroids to settle; max shift=%.4f", shift)
	}

	// Compute cluster means of assignments to ensure separation.
	clusterCounts := make([]int, 2)
	sum := [][]float64{{0, 0}, {0, 0}}
	for idx, a := range assignments {
		clusterCounts[a]++
		for d, v := range items[idx].Vector {
			sum[a][d] += v
		}
	}

	// Both clusters should have at least one point.
	for i, c := range clusterCounts {
		if c == 0 {
			t.Fatalf("cluster %d has zero assignments", i)
		}
	}

	// Each centroid should align closely with one axis.
	for i, c := range centroids {
		var axis int
		if math.Abs(c[0]) > math.Abs(c[1]) {
			axis = 0
		} else {
			axis = 1
		}
		if math.Abs(c[axis]) < 0.9 {
			t.Fatalf("centroid %d not well-aligned with an axis: %v", i, c)
		}
	}
}

func TestNormalizeClusterItemsDropsZeroVectors(t *testing.T) {
	items := []ClusterItem{
		{ID: 1, Vector: []float64{0, 0}},
		{ID: 2, Vector: []float64{3, 4}},
	}
	out := normalizeClusterItems(items)
	if len(out) != 1 {
		t.Fatalf("expected 1 item after dropping zeros, got %d", len(out))
	}
	if out[0].ID != 2 {
		t.Fatalf("unexpected ID kept: %d", out[0].ID)
	}
	norm := l2Norm(out[0].Vector)
	if math.Abs(norm-1) > 1e-9 {
		t.Fatalf("normalized vector not unit length: norm=%.12f", norm)
	}
}
