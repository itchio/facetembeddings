package main

import (
	"errors"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sync"

	"github.com/james-bowman/sparse"
	"gonum.org/v1/gonum/mat"
)

// ALSConfig holds ALS-specific hyperparameters.
type ALSConfig struct {
	Rank           int     // Latent factor dimension
	MaxIterations  int     // Maximum number of iterations
	Lambda         float64 // Regularization parameter
	ConvergenceEps float64 // Early stopping threshold
}

// Converts a SymDense matrix to CSR format using james-bowman/sparse.
// For symmetric input we only materialize the upper triangle (including diagonal)
// and rely on symmetry during factor updates to avoid doubling NNZ and work.
func BuildSparseCSR(sym *mat.SymDense) *sparse.CSR {
	n := sym.SymmetricDim()
	dok := sparse.NewDOK(n, n)

	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			v := sym.At(i, j)
			if v != 0 {
				dok.Set(i, j, v)
				if i != j {
					dok.Set(j, i, v)
				}
			}
		}
	}

	return dok.ToCSR()
}

// Performs ALS matrix factorization on a symmetric matrix.
// For symmetric matrix V (e.g., tag co-occurrence), factorizes as V ≈ W × W^T.
func ComputeALSEmbeddings(co *mat.SymDense, vocab Vocabulary, cfg ALSConfig) (map[string][]float64, error) {
	n := co.SymmetricDim()
	k := cfg.Rank

	if k <= 0 || k > n {
		return nil, errors.New("invalid rank for ALS: must be positive and <= matrix dimension")
	}

	// Convert to sparse CSR format for memory-efficient access
	sparseV := BuildSparseCSR(co)
	nnz := sparseV.NNZ()
	sparsity := 1.0 - float64(nnz)/float64(n*n)
	log.Printf("ALS: matrix dimension=%d, non-zeros=%d, sparsity=%.2f%%", n, nnz, sparsity*100)

	// Initialize W randomly with small values
	W := mat.NewDense(n, k, nil)
	initializeRandomMatrix(W, 42)

	// Precompute λI for regularization
	lambdaI := mat.NewDense(k, k, nil)
	for i := 0; i < k; i++ {
		lambdaI.Set(i, i, cfg.Lambda)
	}

	numWorkers := runtime.NumCPU()
	log.Printf("ALS: using %d workers for parallel updates", numWorkers)

	prevLoss := math.MaxFloat64

	// Build a fixed random sample of non-zero entries for consistent loss reporting.
	// Sampling upfront avoids iterating the full CSR each iteration.
	sampleRows, sampleCols, sampleVals := sampleCSR(sparseV, 10_000, 42)

	for iter := 0; iter < cfg.MaxIterations; iter++ {
		// Update all rows of W in parallel
		W_new := updateFactorsParallel(W, sparseV, k, cfg.Lambda, numWorkers)

		// Copy W_new back to W
		W.Copy(W_new)

		// Compute reconstruction loss on fixed sample for convergence check
		loss := computeFixedSampleLoss(W, sampleRows, sampleCols, sampleVals)

		relChange := math.Abs(prevLoss-loss) / math.Max(1.0, math.Abs(prevLoss))
		log.Printf("ALS iteration %d/%d: loss=%.6f, relative_change=%.6f", iter+1, cfg.MaxIterations, loss, relChange)

		// Check convergence
		if relChange < cfg.ConvergenceEps {
			log.Printf("ALS converged at iteration %d (relative change %.2e < %.2e)", iter+1, relChange, cfg.ConvergenceEps)
			break
		}
		prevLoss = loss
	}

	// Extract embeddings from W
	return extractEmbeddingsFromW(W, vocab), nil
}

// Updates all rows of W in parallel using double buffering.
func updateFactorsParallel(W *mat.Dense, V *sparse.CSR, k int, lambda float64, numWorkers int) *mat.Dense {
	n, _ := W.Dims()
	W_new := mat.NewDense(n, k, nil)

	var wg sync.WaitGroup
	rowChan := make(chan int, n)

	// Spawn workers
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Each worker owns its buffers to avoid per-row allocations.
			lambdaI := mat.NewDense(k, k, nil)
			for i := 0; i < k; i++ {
				lambdaI.Set(i, i, lambda)
			}
			var WJ mat.Dense
			var WtW mat.Dense
			WtW.ReuseAs(k, k)
			indices := make([]int, 0, 64)
			values := make([]float64, 0, 64)

			for i := range rowChan {
				indices = indices[:0]
				values = values[:0]
				updateSingleRow(W, W_new, V, i, k, lambdaI, &WJ, &WtW, &indices, &values)
			}
		}()
	}

	// Send work
	for i := 0; i < n; i++ {
		rowChan <- i
	}
	close(rowChan)

	wg.Wait()
	return W_new
}

// Updates row i of W using the closed-form ALS solution.
// Reads from W (old values), writes to W_new.
func updateSingleRow(W, W_new *mat.Dense, V *sparse.CSR, i, k int, lambdaI, WJ, WtW *mat.Dense, indices *[]int, values *[]float64) {
	// Collect non-zero entries from row i
	V.DoRowNonZero(i, func(_, j int, v float64) {
		*indices = append(*indices, j)
		*values = append(*values, v)
	})

	numObs := len(*indices)
	if numObs == 0 {
		// No observations: set to zeros
		for d := 0; d < k; d++ {
			W_new.Set(i, d, 0)
		}
		return
	}

	// Build W_J (rows of W corresponding to non-zero columns)
	WJ.Reset()
	WJ.ReuseAs(numObs, k)
	for idx, j := range *indices {
		for d := 0; d < k; d++ {
			WJ.Set(idx, d, W.At(j, d))
		}
	}

	// Compute W_J^T × W_J + λI
	WtW.Zero()
	WtW.Mul(WJ.T(), WJ)
	WtW.Add(WtW, lambdaI)

	// Compute W_J^T × v_i
	vi := mat.NewVecDense(numObs, *values)
	Wtv := mat.NewVecDense(k, nil)
	Wtv.MulVec(WJ.T(), vi)

	// Solve (W_J^T × W_J + λI) × w_i = W_J^T × v_i using LU decomposition
	var solution mat.VecDense
	var lu mat.LU
	lu.Factorize(WtW)
	if err := lu.SolveVecTo(&solution, false, Wtv); err != nil {
		// Keep zeros on failure
		return
	}
	for d := 0; d < k; d++ {
		W_new.Set(i, d, solution.AtVec(d))
	}
}

// sampleCSR selects up to maxSamples non-zero entries from a CSR matrix using reservoir sampling.
// Returns slices of row indices, column indices, and values. Seeded for repeatability.
func sampleCSR(V *sparse.CSR, maxSamples int, seed int64) ([]int, []int, []float64) {
	raw := V.RawMatrix()
	rows := make([]int, 0, maxSamples)
	cols := make([]int, 0, maxSamples)
	vals := make([]float64, 0, maxSamples)

	rng := rand.New(rand.NewSource(seed))
	count := 0

	for i := 0; i < len(raw.Indptr)-1; i++ {
		for idx := raw.Indptr[i]; idx < raw.Indptr[i+1]; idx++ {
			j := raw.Ind[idx]
			vij := raw.Data[idx]
			count++

			if len(rows) < maxSamples {
				rows = append(rows, i)
				cols = append(cols, j)
				vals = append(vals, vij)
				continue
			}

			// Reservoir sampling replacement
			r := rng.Intn(count)
			if r < maxSamples {
				rows[r] = i
				cols[r] = j
				vals[r] = vij
			}
		}
	}

	return rows, cols, vals
}

// computeFixedSampleLoss computes reconstruction loss on a fixed set of entries.
func computeFixedSampleLoss(W *mat.Dense, rows, cols []int, vals []float64) float64 {
	_, k := W.Dims()
	if len(rows) == 0 {
		return 0
	}

	var loss float64
	for idx := range rows {
		i := rows[idx]
		j := cols[idx]
		vij := vals[idx]

		var dot float64
		for d := 0; d < k; d++ {
			dot += W.At(i, d) * W.At(j, d)
		}

		diff := vij - dot
		loss += diff * diff
	}

	return loss / float64(len(rows))
}

// Fills a matrix with small random values for initialization.
func initializeRandomMatrix(m *mat.Dense, seed int64) {
	r, c := m.Dims()
	rng := rand.New(rand.NewSource(seed))
	scale := 0.01

	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			m.Set(i, j, rng.NormFloat64()*scale)
		}
	}
}

// Converts the W factor matrix to an embeddings map keyed by tag name.
func extractEmbeddingsFromW(W *mat.Dense, vocab Vocabulary) map[string][]float64 {
	n, k := W.Dims()
	embeddings := make(map[string][]float64, n)

	for i := 0; i < n; i++ {
		vec := make([]float64, k)
		for j := 0; j < k; j++ {
			vec[j] = W.At(i, j)
		}
		embeddings[vocab.IndexToTag[i]] = vec
	}

	return embeddings
}
