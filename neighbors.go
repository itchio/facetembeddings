package main

import (
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
)

// PrintNeighbors logs the top-k nearest facets (by cosine similarity) for
// each requested facet. Used as a post-training spot check: if the neighbor
// lists for familiar tags look wrong, the run shouldn't ship. Cosine ignores
// the SIF weight baked into vector magnitudes, so this inspects pure
// direction.
func PrintNeighbors(embeddings map[string][]float64, facets []string, k int) {
	type scored struct {
		facet string
		score float64
	}

	for _, facet := range facets {
		facet = strings.TrimSpace(facet)
		if facet == "" {
			continue
		}

		query, ok := embeddings[facet]
		if !ok {
			log.Printf("neighbors of %s: (not in vocabulary)", facet)
			continue
		}

		var results []scored
		for other, vec := range embeddings {
			if other == facet {
				continue
			}
			results = append(results, scored{other, cosine(query, vec)})
		}

		sort.Slice(results, func(i, j int) bool { return results[i].score > results[j].score })
		if len(results) > k {
			results = results[:k]
		}

		parts := make([]string, len(results))
		for i, r := range results {
			parts[i] = fmt.Sprintf("%s %.3f", r.facet, r.score)
		}
		log.Printf("neighbors of %s: %s", facet, strings.Join(parts, ", "))
	}
}

func cosine(a, b []float64) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
