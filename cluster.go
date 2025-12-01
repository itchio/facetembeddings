package main

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"
	"time"

	"github.com/lib/pq"
)

// ClusterConfig controls clustering behavior.
type ClusterConfig struct {
	ClusterTable   string
	ClusterColumn  string
	NumClusters    int
	MaxIterations  int
	Tolerance      float64
	Seed           int64
	ExamplesPerRow int // number of example IDs to print per cluster
}

// ClusterItem is a single row fetched for clustering.
type ClusterItem struct {
	ID     int64
	Vector []float64
}

// RunClusterCommand loads vectors from a table and clusters them using spherical k-means.
func RunClusterCommand(ctx context.Context, db *sql.DB, cfg ClusterConfig) error {
	if cfg.ClusterTable == "" {
		return errors.New("cluster table is required (--cluster-table)")
	}
	if cfg.ClusterColumn == "" {
		cfg.ClusterColumn = "vector"
	}
	if cfg.NumClusters <= 1 {
		return fmt.Errorf("number of clusters must be > 1 (got %d)", cfg.NumClusters)
	}
	if cfg.MaxIterations <= 0 {
		cfg.MaxIterations = 50
	}
	if cfg.Tolerance <= 0 {
		cfg.Tolerance = 1e-4
	}
	if cfg.ExamplesPerRow <= 0 {
		cfg.ExamplesPerRow = 5
	}
	if cfg.Seed == 0 {
		cfg.Seed = time.Now().UnixNano()
	}

	log.Printf("loading vectors from table=%q column=%q...", cfg.ClusterTable, cfg.ClusterColumn)
	items, dim, err := LoadClusterItems(ctx, db, cfg.ClusterTable, cfg.ClusterColumn)
	if err != nil {
		return err
	}
	if len(items) == 0 {
		return errors.New("no vectors loaded for clustering")
	}
	if cfg.NumClusters > len(items) {
		return fmt.Errorf("requested clusters (%d) exceeds number of items (%d)", cfg.NumClusters, len(items))
	}
	log.Printf("loaded %d vectors (dim=%d); filtering/normalizing...", len(items), dim)

	normItems := normalizeClusterItems(items)
	if len(normItems) == 0 {
		return errors.New("all vectors were zero-length after normalization")
	}

	log.Printf("running spherical k-means: k=%d, max_iter=%d, tol=%.2e, seed=%d", cfg.NumClusters, cfg.MaxIterations, cfg.Tolerance, cfg.Seed)
	assignments, centroids, iter, shift, err := SphericalKMeans(normItems, cfg.NumClusters, cfg.MaxIterations, cfg.Tolerance, cfg.Seed)
	if err != nil {
		return err
	}

	log.Printf("k-means converged in %d iterations (max centroid shift=%.4f)", iter, shift)
	printClusters(normItems, assignments, centroids, cfg.ExamplesPerRow)
	return nil
}

// LoadClusterItems queries a table for id + vector column.
func LoadClusterItems(ctx context.Context, db *sql.DB, table, column string) ([]ClusterItem, int, error) {
	query := fmt.Sprintf(`SELECT id, %s FROM %s ORDER BY id`, quoteIdentifier(column), quoteIdentifier(table))

	rows, err := db.QueryContext(ctx, query)
	if err != nil {
		return nil, 0, fmt.Errorf("query cluster items: %w", err)
	}
	defer rows.Close()

	var (
		items         []ClusterItem
		expectedDim   int
		skippedZero   int
		skippedLength int
	)

	for rows.Next() {
		var item ClusterItem
		var vec pq.Float64Array
		if err := rows.Scan(&item.ID, &vec); err != nil {
			return nil, 0, fmt.Errorf("scan cluster item: %w", err)
		}
		item.Vector = []float64(vec)

		if len(item.Vector) == 0 {
			skippedZero++
			continue
		}

		if expectedDim == 0 {
			expectedDim = len(item.Vector)
		} else if len(item.Vector) != expectedDim {
			skippedLength++
			continue
		}

		items = append(items, item)
	}

	if err := rows.Err(); err != nil {
		return nil, 0, fmt.Errorf("iterate cluster items: %w", err)
	}

	if skippedZero > 0 {
		log.Printf("skipped %d items with empty vectors", skippedZero)
	}
	if skippedLength > 0 {
		log.Printf("skipped %d items with mismatched vector length", skippedLength)
	}

	return items, expectedDim, nil
}

// normalizeClusterItems returns items with unit-length vectors, dropping zero vectors.
func normalizeClusterItems(items []ClusterItem) []ClusterItem {
	out := make([]ClusterItem, 0, len(items))
	for _, it := range items {
		norm := l2Norm(it.Vector)
		if norm == 0 {
			continue
		}
		vec := make([]float64, len(it.Vector))
		for i, v := range it.Vector {
			vec[i] = v / norm
		}
		out = append(out, ClusterItem{ID: it.ID, Vector: vec})
	}
	return out
}

// SphericalKMeans runs k-means using cosine similarity on unit vectors.
func SphericalKMeans(items []ClusterItem, k, maxIter int, tol float64, seed int64) ([]int, [][]float64, int, float64, error) {
	if k <= 1 {
		return nil, nil, 0, 0, errors.New("k must be > 1")
	}
	if len(items) < k {
		return nil, nil, 0, 0, errors.New("not enough items for requested clusters")
	}

	rng := rand.New(rand.NewSource(seed))
	centroids := initCentroids(items, k, rng)
	assignments := make([]int, len(items))
	prevShift := math.MaxFloat64

	for iter := 1; iter <= maxIter; iter++ {
		// Assignment step
		for i, it := range items {
			assignments[i] = closestCentroid(it.Vector, centroids)
		}

		// Update step
		newCentroids := make([][]float64, k)
		counts := make([]int, k)
		for i := 0; i < k; i++ {
			newCentroids[i] = make([]float64, len(centroids[i]))
		}

		for idx, it := range items {
			c := assignments[idx]
			counts[c]++
			acc := newCentroids[c]
			for d, v := range it.Vector {
				acc[d] += v
			}
		}

		// Handle empty clusters by re-seeding them with random points.
		for i := 0; i < k; i++ {
			if counts[i] == 0 {
				reseed := items[rng.Intn(len(items))].Vector
				copy(newCentroids[i], reseed)
				counts[i] = 1
			}
		}

		// Normalize centroids and compute shift
		var maxShift float64
		for i := 0; i < k; i++ {
			norm := l2Norm(newCentroids[i])
			if norm == 0 {
				continue
			}
			for d := range newCentroids[i] {
				newCentroids[i][d] /= norm
			}
			// On the unit sphere, use cosine similarity to gauge movement.
			diff := 1 - dot(centroids[i], newCentroids[i])
			if diff > maxShift {
				maxShift = diff
			}
		}

		centroids = newCentroids

		if maxShift < tol {
			return assignments, centroids, iter, maxShift, nil
		}
		prevShift = maxShift
	}

	return assignments, centroids, maxIter, prevShift, nil
}

func initCentroids(items []ClusterItem, k int, rng *rand.Rand) [][]float64 {
	centroids := make([][]float64, 0, k)
	seen := make(map[int]struct{}, k)
	for len(centroids) < k {
		idx := rng.Intn(len(items))
		if _, ok := seen[idx]; ok {
			continue
		}
		seen[idx] = struct{}{}
		vecCopy := make([]float64, len(items[idx].Vector))
		copy(vecCopy, items[idx].Vector)
		centroids = append(centroids, vecCopy)
	}
	return centroids
}

func closestCentroid(vec []float64, centroids [][]float64) int {
	best := 0
	bestScore := -1.0
	for i, c := range centroids {
		score := dot(vec, c) // cosine similarity on unit vectors
		if score > bestScore {
			bestScore = score
			best = i
		}
	}
	return best
}

func dot(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func l2Norm(vec []float64) float64 {
	var sum float64
	for _, v := range vec {
		sum += v * v
	}
	return math.Sqrt(sum)
}

// printClusters prints a text summary of clusters ordered by size.
func printClusters(items []ClusterItem, assignments []int, centroids [][]float64, examples int) {
	type clusterInfo struct {
		ID     int
		Size   int
		IDs    []int64
		Center []float64
	}

	k := len(centroids)
	clusters := make([]clusterInfo, k)
	for i := 0; i < k; i++ {
		clusters[i] = clusterInfo{
			ID:     i,
			IDs:    make([]int64, 0, examples),
			Center: centroids[i],
		}
	}

	for idx, assignment := range assignments {
		info := &clusters[assignment]
		info.Size++
		if len(info.IDs) < examples {
			info.IDs = append(info.IDs, items[idx].ID)
		}
	}

	sort.Slice(clusters, func(i, j int) bool {
		return clusters[i].Size > clusters[j].Size
	})

	log.Printf("cluster summary (top %d example IDs per cluster):", examples)
	for _, c := range clusters {
		log.Printf("  cluster %d: size=%d, examples=%v", c.ID, c.Size, c.IDs)
	}
}
