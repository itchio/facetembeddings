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

// accumulates tag co-occurrence counts in a dense symmetric matrix.
func BuildCooccurrenceMatrix(items []ItemTags, vocab Vocabulary) *mat.SymDense {
	n := len(vocab.IndexToTag)
	co := mat.NewSymDense(n, nil)

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

		for i := 0; i < len(indexBuf); i++ {
			for j := i; j < len(indexBuf); j++ {
				a := indexBuf[i]
				b := indexBuf[j]
				co.SetSym(a, b, co.At(a, b)+1)
			}
		}
	}

	return co
}

// computes the Positive Pointwise Mutual Information (PPMI) matrix.
func BuildPPMIMatrix(items []ItemTags, vocab Vocabulary, minCooccurrence int) *mat.SymDense {
	n := len(vocab.IndexToTag)
	co := mat.NewSymDense(n, nil)

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

		// Count co-occurrences for pairs of unique tags in an item
		for i := 0; i < len(indexBuf); i++ {
			for j := i + 1; j < len(indexBuf); j++ {
				a := indexBuf[i]
				b := indexBuf[j]
				co.SetSym(a, b, co.At(a, b)+1)
			}
		}
	}

	ApplyMinCooccurrence(co, minCooccurrence)

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

	totalTagObservations := totalPairs * 2.0
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
			p_i := tagCounts[i] / totalTagObservations
			p_j := tagCounts[j] / totalTagObservations

			if p_i == 0 || p_j == 0 {
				continue
			}

			pmi := math.Log2(p_ij / (p_i * p_j))
			ppmi.SetSym(i, j, math.Max(0, pmi))
		}
	}

	return ppmi
}

// removes low-signal entries.
func ApplyMinCooccurrence(co *mat.SymDense, minCount int) {
	if minCount <= 1 {
		return
	}

	n := co.SymmetricDim()
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			if co.At(i, j) < float64(minCount) {
				co.SetSym(i, j, 0)
			}
		}
	}
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

// executes the entire embedding workflow.
func RunEmbeddingPipeline(items []ItemTags, cfg EmbeddingConfig) (map[string][]float64, Vocabulary, error) {
	if cfg.EmbeddingDim <= 0 {
		return nil, Vocabulary{}, fmt.Errorf("embedding dim must be positive (got %d)", cfg.EmbeddingDim)
	}
	if cfg.MinTagFrequency <= 0 {
		cfg.MinTagFrequency = 1
	}

	log.Printf("building vocabulary (min freq=%d, max tags=%d)...", cfg.MinTagFrequency, cfg.MaxTags)
	vocabStart := time.Now()
	vocab, err := BuildVocabulary(items, cfg)
	if err != nil {
		return nil, Vocabulary{}, err
	}
	log.Printf("built vocabulary of %d tags in %s", len(vocab.IndexToTag), time.Since(vocabStart).Round(time.Millisecond))

	var co *mat.SymDense
	matrixStart := time.Now()

	switch cfg.MatrixType {
	case "cooc":
		log.Printf("building co-occurrence matrix...")
		co = BuildCooccurrenceMatrix(items, vocab)
		ApplyMinCooccurrence(co, cfg.MinCooccurrence)
		log.Printf("co-occurrence matrix ready in %s", time.Since(matrixStart).Round(time.Millisecond))
	case "ppmi":
		log.Printf("building PPMI matrix...")
		co = BuildPPMIMatrix(items, vocab, cfg.MinCooccurrence)
		log.Printf("PPMI matrix ready in %s", time.Since(matrixStart).Round(time.Millisecond))
	default:
		return nil, Vocabulary{}, fmt.Errorf("unknown matrix type: %q", cfg.MatrixType)
	}

	log.Printf("running SVD (dim=%d)...", cfg.EmbeddingDim)
	svdStart := time.Now()
	embeddings, err := ComputeEmbeddings(co, vocab, cfg.EmbeddingDim)
	if err != nil {
		return nil, Vocabulary{}, err
	}
	log.Printf("svd complete in %s", time.Since(svdStart).Round(time.Millisecond))

	log.Printf("normalizing embeddings...")
	normStart := time.Now()
	NormalizeEmbeddings(embeddings)
	log.Printf("normalization complete in %s", time.Since(normStart).Round(time.Millisecond))

	return embeddings, vocab, nil
}
