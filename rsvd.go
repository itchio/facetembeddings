package main

import (
	"errors"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat"
)

const (
	rsvdOversample = 128
	rsvdPowerIters = 6
	rsvdSeed       = 42 // fixed for reproducible builds
)

// ComputeRSVDEmbeddings computes a randomized truncated SVD of the sparse
// symmetric matrix and derives embeddings the same way as the exact path
// (u_i * sqrt(sigma)). Cost is O(nnz * (dim+oversample)) per pass instead of
// the dense SVD's O(n^3), which makes large vocabularies train in minutes.
//
// Standard Halko/Martinsson/Tropp scheme: sketch the range with a gaussian
// test matrix, refine with power iterations (re-orthonormalizing between
// passes for stability), then solve a small exact SVD in the sketched basis.
func ComputeRSVDEmbeddings(a *csrMatrix, vocab Vocabulary, dim int) (map[string][]float64, error) {
	if dim <= 0 {
		return nil, errors.New("embedding dim must be positive")
	}
	n := a.n
	if dim > n {
		dim = n
	}
	m := dim + rsvdOversample
	if m > n {
		m = n
	}

	// gaussian sketch
	rng := rand.New(rand.NewSource(rsvdSeed))
	omega := mat.NewDense(n, m, nil)
	{
		raw := omega.RawMatrix()
		for i := range raw.Data {
			raw.Data[i] = rng.NormFloat64()
		}
	}

	// Q = orth(A * Omega), refined by power iterations (A is symmetric)
	y := mulCSRDense(a, omega)
	q, err := orthonormalize(y)
	if err != nil {
		return nil, err
	}
	for it := 0; it < rsvdPowerIters; it++ {
		y = mulCSRDense(a, q)
		q, err = orthonormalize(y)
		if err != nil {
			return nil, err
		}
	}

	// B = Q^T A = (A Q)^T by symmetry; small exact SVD of B (m x n)
	aq := mulCSRDense(a, q) // n x m
	var b mat.Dense
	b.CloneFrom(aq.T())

	var svd mat.SVD
	if ok := svd.Factorize(&b, mat.SVDThin); !ok {
		return nil, errors.New("rsvd: small SVD factorization failed")
	}
	values := svd.Values(nil)
	var ub mat.Dense
	svd.UTo(&ub) // m x m

	// U = Q * U_B, take leading dim columns scaled by sqrt(sigma)
	var u mat.Dense
	u.Mul(q, &ub) // n x m

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

// orthonormalize returns an orthonormal basis for the range of y (n x m,
// n >= m) via thin SVD, which is numerically robust and avoids gonum QR's
// full n x n Q extraction.
func orthonormalize(y *mat.Dense) (*mat.Dense, error) {
	var svd mat.SVD
	if ok := svd.Factorize(y, mat.SVDThin); !ok {
		return nil, errors.New("rsvd: orthonormalization SVD failed")
	}
	var u mat.Dense
	svd.UTo(&u)
	return &u, nil
}

// mulCSRDense computes A * X for sparse symmetric A (n x n) and dense X
// (n x m), parallelized across row chunks.
func mulCSRDense(a *csrMatrix, x *mat.Dense) *mat.Dense {
	start := time.Now()
	n := a.n
	_, m := x.Dims()
	out := mat.NewDense(n, m, nil)

	xRaw := x.RawMatrix()
	outRaw := out.RawMatrix()

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
				outRow := outRaw.Data[i*outRaw.Stride : i*outRaw.Stride+m]
				for p := a.rowP[i]; p < a.rowP[i+1]; p++ {
					j := int(a.cols[p])
					v := a.vals[p]
					xRow := xRaw.Data[j*xRaw.Stride : j*xRaw.Stride+m]
					for k := 0; k < m; k++ {
						outRow[k] += v * xRow[k]
					}
				}
			}
		}(lo, hi)
	}
	wg.Wait()

	log.Printf("rsvd: sparse multiply (%d x %d) in %s", n, m, time.Since(start).Round(time.Millisecond))
	return out
}
