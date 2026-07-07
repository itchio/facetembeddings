package main

import (
	"fmt"
	"log"
	"math"
	"runtime"
	"sort"
	"sync"
)

// csrMatrix is a minimal symmetric CSR representation used by the randomized
// SVD path. Unlike the dense SymDense pipeline, memory scales with the number
// of non-zero tag pairs rather than vocabulary squared, which is what makes
// large vocabularies tractable.
type csrMatrix struct {
	n    int
	rowP []int64   // n+1 row pointers into cols/vals
	cols []int32   // column index per non-zero
	vals []float64 // value per non-zero
}

func (m *csrMatrix) nnz() int { return len(m.cols) }

// pairKey packs an upper-triangle pair (a <= b) into a map key.
func pairKey(a, b int) uint64 {
	if a > b {
		a, b = b, a
	}
	return uint64(a)<<32 | uint64(b)
}

type pairEntry struct {
	w float64 // quality-weighted, per-game-normalized mass
	n uint32  // raw game count (for MinCooccurrence pruning)
}

// BuildSparseMatrix accumulates the co-occurrence matrix in sparse form and
// applies the same transforms as the dense builders: quality weighting and
// per-game normalization via itemIncrement, raw-game-count pruning via
// MinCooccurrence, and (for MatrixType "ppmi") the smoothed PPMI transform
// mirroring BuildPPMIMatrix exactly. MatrixType "cooc" keeps weighted counts
// and, matching the dense builder, includes the diagonal.
func BuildSparseMatrix(items []ItemTags, vocab Vocabulary, cfg EmbeddingConfig) (*csrMatrix, error) {
	n := len(vocab.IndexToTag)
	includeDiagonal := cfg.MatrixType == "cooc"

	pairs := make(map[uint64]pairEntry, 1<<20)

	indexBuf := make([]int, 0, 64)
	seen := make(map[int]struct{}, 64)

	for _, item := range items {
		indexBuf = indexBuf[:0]
		for _, tag := range item.Tags {
			idx, ok := vocab.TagToIndex[tag]
			if !ok {
				continue
			}
			if _, found := seen[idx]; found {
				continue
			}
			seen[idx] = struct{}{}
			indexBuf = append(indexBuf, idx)
		}
		for _, idx := range indexBuf {
			delete(seen, idx)
		}

		minTags := 2
		if includeDiagonal {
			minTags = 1
		}
		if len(indexBuf) < minTags {
			continue
		}

		inc := itemIncrement(item, len(indexBuf), cfg.PerGameNorm)

		for i := 0; i < len(indexBuf); i++ {
			jStart := i + 1
			if includeDiagonal {
				jStart = i
			}
			for j := jStart; j < len(indexBuf); j++ {
				k := pairKey(indexBuf[i], indexBuf[j])
				e := pairs[k]
				e.w += inc
				e.n++
				pairs[k] = e
			}
		}
	}

	// prune on raw game counts, matching rawCounter semantics
	if cfg.MinCooccurrence > 1 {
		for k, e := range pairs {
			if int(e.n) < cfg.MinCooccurrence {
				delete(pairs, k)
			}
		}
	}

	if len(pairs) == 0 {
		return nil, fmt.Errorf("sparse matrix has no entries")
	}
	log.Printf("sparse accumulation: %d distinct pairs", len(pairs))

	switch cfg.MatrixType {
	case "ppmi":
		applySparsePPMI(pairs, n, cfg.PPMIAlpha)
	case "cooc":
		// weighted counts as-is
	default:
		return nil, fmt.Errorf("unknown matrix type: %q", cfg.MatrixType)
	}

	return pairsToCSR(pairs, n), nil
}

// applySparsePPMI replaces pair weights with smoothed PPMI values, mirroring
// the dense BuildPPMIMatrix math: diagonal excluded, marginals from row sums,
// context distribution smoothed with alpha, negatives clipped to zero
// (dropped from the sparse map).
func applySparsePPMI(pairs map[uint64]pairEntry, n int, alpha float64) {
	if alpha <= 0 {
		alpha = 1
	}

	var totalPairs float64
	tagCounts := make([]float64, n)
	for k, e := range pairs {
		a, b := int(k>>32), int(k&0xffffffff)
		if a == b {
			// dense PPMI ignores the diagonal; keep parity
			delete(pairs, k)
			continue
		}
		totalPairs += e.w
		tagCounts[a] += e.w
		tagCounts[b] += e.w
	}
	if totalPairs == 0 {
		return
	}

	var smoothedTotal float64
	smoothedCounts := make([]float64, n)
	for i, c := range tagCounts {
		smoothedCounts[i] = math.Pow(c, alpha)
		smoothedTotal += smoothedCounts[i]
	}

	for k, e := range pairs {
		a, b := int(k>>32), int(k&0xffffffff)

		p_ij := e.w / totalPairs
		p_a := smoothedCounts[a] / smoothedTotal
		p_b := smoothedCounts[b] / smoothedTotal
		if p_a == 0 || p_b == 0 {
			delete(pairs, k)
			continue
		}

		pmi := math.Log2(p_ij / (p_a * p_b))
		if pmi <= 0 {
			delete(pairs, k)
			continue
		}
		e.w = pmi
		pairs[k] = e
	}
}

// pairsToCSR expands upper-triangle pairs into a symmetric CSR matrix.
func pairsToCSR(pairs map[uint64]pairEntry, n int) *csrMatrix {
	rowCounts := make([]int64, n+1)
	for k := range pairs {
		a, b := int(k>>32), int(k&0xffffffff)
		rowCounts[a+1]++
		if a != b {
			rowCounts[b+1]++
		}
	}
	for i := 0; i < n; i++ {
		rowCounts[i+1] += rowCounts[i]
	}

	nnz := rowCounts[n]
	m := &csrMatrix{
		n:    n,
		rowP: rowCounts,
		cols: make([]int32, nnz),
		vals: make([]float64, nnz),
	}

	cursor := make([]int64, n)
	for k, e := range pairs {
		a, b := int(k>>32), int(k&0xffffffff)
		pos := m.rowP[a] + cursor[a]
		m.cols[pos] = int32(b)
		m.vals[pos] = e.w
		cursor[a]++
		if a != b {
			pos = m.rowP[b] + cursor[b]
			m.cols[pos] = int32(a)
			m.vals[pos] = e.w
			cursor[b]++
		}
	}

	// sort columns within each row for deterministic layout
	var wg sync.WaitGroup
	workers := runtime.NumCPU()
	chunk := (n + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo, hi := w*chunk, min((w+1)*chunk, n)
		if lo >= hi {
			continue
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for i := lo; i < hi; i++ {
				s, e := m.rowP[i], m.rowP[i+1]
				row := rowView{m.cols[s:e], m.vals[s:e]}
				sort.Sort(row)
			}
		}(lo, hi)
	}
	wg.Wait()

	return m
}

type rowView struct {
	cols []int32
	vals []float64
}

func (r rowView) Len() int           { return len(r.cols) }
func (r rowView) Less(i, j int) bool { return r.cols[i] < r.cols[j] }
func (r rowView) Swap(i, j int) {
	r.cols[i], r.cols[j] = r.cols[j], r.cols[i]
	r.vals[i], r.vals[j] = r.vals[j], r.vals[i]
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
