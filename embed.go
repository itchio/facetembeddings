package main

import (
	"errors"
	"fmt"
	"log"
	"math"
	"sort"
	"time"

	"gonum.org/v1/gonum/mat"
)

// EmbeddingConfig controls how embeddings are generated.
type EmbeddingConfig struct {
	EmbeddingDim    int
	MinTagFrequency int
	MaxTags         int
	MinCooccurrence int
	MatrixType      string

	// SIF pooling-weight parameter (a in a/(a+p)). Each output vector is
	// scaled by its facet's weight so a plain average of vectors downstream
	// becomes a frequency-weighted average. <= 0 disables weighting
	// (vectors stay unit length).
	SIFParam float64

	// PPMI context-distribution smoothing exponent (Levy et al. 2015).
	// Marginal probabilities are computed from counts^alpha, which damps
	// PMI's bias toward rare tags. 1 reproduces classic PPMI.
	PPMIAlpha float64

	// When set, each item's pair increments are divided by (numTags-1) so
	// every game contributes total co-occurrence mass proportional to its
	// tag count rather than its square (prevents heavily-tagged pages from
	// dominating the matrix).
	PerGameNorm bool

	// Factorization method: "svd" or "als"
	FactorizationType string

	// ALS-specific parameters
	ALSIterations     int     // Maximum number of ALS iterations
	ALSRegularization float64 // Lambda regularization term
	ALSConvergence    float64 // Convergence threshold for early stopping
}

// Vocabulary maps tags to indices and exposes metadata.
type Vocabulary struct {
	TagToIndex map[string]int
	IndexToTag []string
	Frequency  map[string]int
}

// counts tag frequency and applies filtering rules.
func BuildVocabulary(items []ItemTags, cfg EmbeddingConfig) (Vocabulary, error) {
	freq := make(map[string]int)
	totalAssignments := 0
	for _, item := range items {
		for _, tag := range item.Tags {
			freq[tag]++
			totalAssignments++
		}
	}
	uniqueTags := len(freq)
	log.Printf("tag corpus: %d total tag assignments across %d unique tags", totalAssignments, uniqueTags)

	type tagFreq struct {
		Tag  string
		Freq int
	}

	var entries []tagFreq
	for tag, count := range freq {
		if count >= cfg.MinTagFrequency {
			entries = append(entries, tagFreq{Tag: tag, Freq: count})
		}
	}

	if len(entries) == 0 {
		return Vocabulary{}, errors.New("no tags passed min frequency filter")
	}

	log.Printf(
		"filtering tags: min frequency %d keeps %d tags (dropped %d)",
		cfg.MinTagFrequency,
		len(entries),
		uniqueTags-len(entries),
	)

	sort.Slice(entries, func(i, j int) bool {
		if entries[i].Freq == entries[j].Freq {
			return entries[i].Tag < entries[j].Tag
		}
		return entries[i].Freq > entries[j].Freq
	})

	if cfg.MaxTags > 0 && len(entries) > cfg.MaxTags {
		dropped := len(entries) - cfg.MaxTags
		entries = entries[:cfg.MaxTags]
		log.Printf("trimming to max %d tags (dropped %d additional tags)", cfg.MaxTags, dropped)
	}

	if len(entries) > 0 {
		top := len(entries)
		if top > 5 {
			top = 5
		}
		log.Printf("top %d tags by frequency:", top)
		for i := 0; i < top; i++ {
			log.Printf("  %s: %d", entries[i].Tag, entries[i].Freq)
		}
	}

	vocab := Vocabulary{
		TagToIndex: make(map[string]int, len(entries)),
		IndexToTag: make([]string, len(entries)),
		Frequency:  make(map[string]int, len(entries)),
	}

	for idx, entry := range entries {
		vocab.TagToIndex[entry.Tag] = idx
		vocab.IndexToTag[idx] = entry.Tag
		vocab.Frequency[entry.Tag] = entry.Freq
	}

	return vocab, nil
}

// itemIncrement computes how much a single item adds to each of its tag
// pairs: its quality weight (1 when weighting is disabled), optionally
// divided by (numTags-1) so total contributed mass grows linearly with tag
// count instead of quadratically.
func itemIncrement(item ItemTags, numTags int, perGameNorm bool) float64 {
	w := item.Weight
	if w <= 0 {
		w = 1
	}
	if perGameNorm && numTags > 1 {
		w /= float64(numTags - 1)
	}
	return w
}

// accumulates tag co-occurrence counts in a dense symmetric matrix.
func BuildCooccurrenceMatrix(items []ItemTags, vocab Vocabulary, cfg EmbeddingConfig) *mat.SymDense {
	n := len(vocab.IndexToTag)
	co := mat.NewSymDense(n, nil)
	rawCo := newRawCounter(n, cfg.MinCooccurrence)

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

		if len(indexBuf) == 0 {
			continue
		}

		inc := itemIncrement(item, len(indexBuf), cfg.PerGameNorm)

		for i := 0; i < len(indexBuf); i++ {
			for j := i; j < len(indexBuf); j++ {
				a := indexBuf[i]
				b := indexBuf[j]
				co.SetSym(a, b, co.At(a, b)+inc)
				rawCo.count(a, b)
			}
		}
	}

	rawCo.prune(co, cfg.MinCooccurrence)

	return co
}

// rawCounter tracks unweighted pair counts (number of games) so that
// MinCooccurrence keeps its intuitive meaning — "the pair must appear on at
// least N games" — independent of quality weighting and per-game
// normalization, which shrink the weighted matrix entries well below 1 per
// game. Only allocated when pruning is actually requested.
type rawCounter struct {
	counts *mat.SymDense
}

func newRawCounter(n, minCooccurrence int) *rawCounter {
	if minCooccurrence <= 1 {
		return &rawCounter{}
	}
	return &rawCounter{counts: mat.NewSymDense(n, nil)}
}

func (r *rawCounter) count(a, b int) {
	if r.counts == nil {
		return
	}
	r.counts.SetSym(a, b, r.counts.At(a, b)+1)
}

// prune zeroes entries of co whose raw game count is below minCount.
func (r *rawCounter) prune(co *mat.SymDense, minCount int) {
	if r.counts == nil || minCount <= 1 {
		return
	}
	n := co.SymmetricDim()
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			if r.counts.At(i, j) < float64(minCount) {
				co.SetSym(i, j, 0)
			}
		}
	}
}

// computes the Positive Pointwise Mutual Information (PPMI) matrix.
// Marginal probabilities are smoothed with cfg.PPMIAlpha (counts^alpha,
// renormalized), which damps PMI's overestimation of associations involving
// rare tags; alpha=1 gives classic PPMI.
func BuildPPMIMatrix(items []ItemTags, vocab Vocabulary, cfg EmbeddingConfig) *mat.SymDense {
	n := len(vocab.IndexToTag)
	co := mat.NewSymDense(n, nil)
	rawCo := newRawCounter(n, cfg.MinCooccurrence)

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

		if len(indexBuf) < 2 {
			continue
		}

		inc := itemIncrement(item, len(indexBuf), cfg.PerGameNorm)

		// Count co-occurrences for pairs of unique tags in an item
		for i := 0; i < len(indexBuf); i++ {
			for j := i + 1; j < len(indexBuf); j++ {
				a := indexBuf[i]
				b := indexBuf[j]
				co.SetSym(a, b, co.At(a, b)+inc)
				rawCo.count(a, b)
			}
		}
	}

	rawCo.prune(co, cfg.MinCooccurrence)

	var totalPairs float64
	for i := 0; i < n; i++ {
		for j := i + 1; j < n; j++ {
			totalPairs += co.At(i, j)
		}
	}

	if totalPairs == 0 {
		return mat.NewSymDense(n, nil)
	}

	tagCounts := make([]float64, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			tagCounts[i] += co.At(i, j)
		}
	}

	alpha := cfg.PPMIAlpha
	if alpha <= 0 {
		alpha = 1
	}

	// Smoothed marginal: p_i = counts_i^alpha / sum_k counts_k^alpha.
	// At alpha=1 this is exactly counts_i / (2 * totalPairs).
	var smoothedTotal float64
	smoothedCounts := make([]float64, n)
	for i := 0; i < n; i++ {
		smoothedCounts[i] = math.Pow(tagCounts[i], alpha)
		smoothedTotal += smoothedCounts[i]
	}

	ppmi := mat.NewSymDense(n, nil)

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			co_ij := co.At(i, j)
			if co_ij == 0 {
				continue
			}

			// Diagonal is zero in this co-occurrence model
			if i == j {
				continue
			}

			p_ij := co_ij / totalPairs
			p_i := smoothedCounts[i] / smoothedTotal
			p_j := smoothedCounts[j] / smoothedTotal

			if p_i == 0 || p_j == 0 {
				continue
			}

			pmi := math.Log2(p_ij / (p_i * p_j))
			ppmi.SetSym(i, j, math.Max(0, pmi))
		}
	}

	return ppmi
}

// runs SVD over the co-occurrence matrix.
func ComputeEmbeddings(co *mat.SymDense, vocab Vocabulary, dim int) (map[string][]float64, error) {
	if dim <= 0 {
		return nil, errors.New("embedding dim must be positive")
	}

	n := co.SymmetricDim()
	if dim > n {
		dim = n
	}

	dense := mat.NewDense(n, n, nil)
	dense.Copy(co)

	var svd mat.SVD
	if ok := svd.Factorize(dense, mat.SVDThin); !ok {
		return nil, errors.New("svd factorization failed")
	}

	values := svd.Values(nil)
	var u mat.Dense
	svd.UTo(&u)

	embeddings := make(map[string][]float64, n)
	for i := 0; i < n; i++ {
		vec := make([]float64, dim)
		for j := 0; j < dim; j++ {
			weight := 0.0
			if j < len(values) {
				weight = math.Sqrt(math.Max(values[j], 0))
			}
			vec[j] = u.At(i, j) * weight
		}
		embeddings[vocab.IndexToTag[i]] = vec
	}

	return embeddings, nil
}

// ComputeFacetWeights returns a SIF-style pooling weight for every vocabulary
// tag: a/(a+p), where p is the fraction of items carrying the tag. a <= 0
// returns weight 1 for every tag. Weights are purely statistical; any product
// policy (eg. minimum influence for monetization/platform facets) is applied
// at the application layer, which can recover the unit vector via the stored
// weight.
func ComputeFacetWeights(vocab Vocabulary, numItems int, a float64) map[string]float64 {
	weights := make(map[string]float64, len(vocab.Frequency))
	for tag, freq := range vocab.Frequency {
		w := 1.0
		if a > 0 && numItems > 0 {
			p := float64(freq) / float64(numItems)
			w = a / (a + p)
		}
		weights[tag] = w
	}
	return weights
}

// scales each (unit) embedding by its pooling weight so a plain average of
// stored vectors downstream becomes a weighted average.
func ApplyFacetWeights(embeddings map[string][]float64, weights map[string]float64) {
	for tag, vec := range embeddings {
		w, ok := weights[tag]
		if !ok || w == 1 {
			continue
		}
		for i := range vec {
			vec[i] *= w
		}
	}
}

// scales all vectors to unit length.
func NormalizeEmbeddings(embeddings map[string][]float64) {
	for tag, vec := range embeddings {
		var sum float64
		for _, v := range vec {
			sum += v * v
		}
		norm := math.Sqrt(sum)
		if norm == 0 {
			continue
		}
		for i := range vec {
			vec[i] /= norm
		}
		embeddings[tag] = vec
	}
}

// executes the entire embedding workflow. Returned vectors are unit length
// scaled by their facet's pooling weight (also returned, keyed by facet).
func RunEmbeddingPipeline(items []ItemTags, cfg EmbeddingConfig) (map[string][]float64, map[string]float64, Vocabulary, error) {
	if cfg.EmbeddingDim <= 0 {
		return nil, nil, Vocabulary{}, fmt.Errorf("embedding dim must be positive (got %d)", cfg.EmbeddingDim)
	}
	if cfg.MinTagFrequency <= 0 {
		cfg.MinTagFrequency = 1
	}
	if cfg.FactorizationType == "" {
		cfg.FactorizationType = "svd"
	}

	log.Printf("building vocabulary (min freq=%d, max tags=%d)...", cfg.MinTagFrequency, cfg.MaxTags)
	vocabStart := time.Now()
	vocab, err := BuildVocabulary(items, cfg)
	if err != nil {
		return nil, nil, Vocabulary{}, err
	}
	log.Printf("built vocabulary of %d tags in %s", len(vocab.IndexToTag), time.Since(vocabStart).Round(time.Millisecond))

	// The rsvd path never materializes the dense matrix: sparse accumulation
	// plus randomized truncated SVD keeps both memory and time proportional
	// to the number of distinct tag pairs rather than vocabulary squared.
	if cfg.FactorizationType == "rsvd" {
		matrixStart := time.Now()
		log.Printf("building sparse %s matrix (alpha=%g, per-game norm=%v)...", cfg.MatrixType, cfg.PPMIAlpha, cfg.PerGameNorm)
		sm, err := BuildSparseMatrix(items, vocab, cfg)
		if err != nil {
			return nil, nil, Vocabulary{}, err
		}
		log.Printf("sparse matrix ready: %d non-zeros in %s", sm.nnz(), time.Since(matrixStart).Round(time.Millisecond))

		factStart := time.Now()
		log.Printf("running randomized SVD (dim=%d, oversample=%d, power iters=%d)...", cfg.EmbeddingDim, rsvdOversample, rsvdPowerIters)
		embeddings, err := ComputeRSVDEmbeddings(sm, vocab, cfg.EmbeddingDim)
		if err != nil {
			return nil, nil, Vocabulary{}, err
		}
		log.Printf("randomized SVD complete in %s", time.Since(factStart).Round(time.Millisecond))

		NormalizeEmbeddings(embeddings)
		weights := ComputeFacetWeights(vocab, len(items), cfg.SIFParam)
		if cfg.SIFParam > 0 {
			log.Printf("scaling vectors by SIF pooling weights (a=%g)...", cfg.SIFParam)
			ApplyFacetWeights(embeddings, weights)
		}
		return embeddings, weights, vocab, nil
	}

	var co *mat.SymDense
	matrixStart := time.Now()

	switch cfg.MatrixType {
	case "cooc":
		log.Printf("building co-occurrence matrix (per-game norm=%v)...", cfg.PerGameNorm)
		co = BuildCooccurrenceMatrix(items, vocab, cfg)
		log.Printf("co-occurrence matrix ready in %s", time.Since(matrixStart).Round(time.Millisecond))
	case "ppmi":
		log.Printf("building PPMI matrix (alpha=%g, per-game norm=%v)...", cfg.PPMIAlpha, cfg.PerGameNorm)
		co = BuildPPMIMatrix(items, vocab, cfg)
		log.Printf("PPMI matrix ready in %s", time.Since(matrixStart).Round(time.Millisecond))
	default:
		return nil, nil, Vocabulary{}, fmt.Errorf("unknown matrix type: %q", cfg.MatrixType)
	}

	var embeddings map[string][]float64
	factStart := time.Now()

	switch cfg.FactorizationType {
	case "svd":
		log.Printf("running SVD (dim=%d)...", cfg.EmbeddingDim)
		embeddings, err = ComputeEmbeddings(co, vocab, cfg.EmbeddingDim)
		if err != nil {
			return nil, nil, Vocabulary{}, err
		}
		log.Printf("SVD complete in %s", time.Since(factStart).Round(time.Millisecond))

	case "als":
		log.Printf("running ALS (dim=%d, iterations=%d, lambda=%.4f, convergence=%.2e)...",
			cfg.EmbeddingDim, cfg.ALSIterations, cfg.ALSRegularization, cfg.ALSConvergence)
		alsCfg := ALSConfig{
			Rank:           cfg.EmbeddingDim,
			MaxIterations:  cfg.ALSIterations,
			Lambda:         cfg.ALSRegularization,
			ConvergenceEps: cfg.ALSConvergence,
		}
		embeddings, err = ComputeALSEmbeddings(co, vocab, alsCfg)
		if err != nil {
			return nil, nil, Vocabulary{}, err
		}
		log.Printf("ALS complete in %s", time.Since(factStart).Round(time.Millisecond))

	default:
		return nil, nil, Vocabulary{}, fmt.Errorf("unknown factorization type: %q (use \"rsvd\", \"svd\" or \"als\")", cfg.FactorizationType)
	}

	log.Printf("normalizing embeddings...")
	normStart := time.Now()
	NormalizeEmbeddings(embeddings)
	log.Printf("normalization complete in %s", time.Since(normStart).Round(time.Millisecond))

	weights := ComputeFacetWeights(vocab, len(items), cfg.SIFParam)
	if cfg.SIFParam > 0 {
		log.Printf("scaling vectors by SIF pooling weights (a=%g)...", cfg.SIFParam)
		ApplyFacetWeights(embeddings, weights)
	} else {
		log.Printf("SIF weighting disabled (a=%g), vectors stay unit length", cfg.SIFParam)
	}

	return embeddings, weights, vocab, nil
}
