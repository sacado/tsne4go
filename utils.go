package tsne4go

// Utility functions for tSNE

import (
	"math"
	"math/rand"
)

// return 0 mean unit standard deviation random number
func gaussRandom() float64 {
	u := 2*rand.Float64() - 1
	v := 2*rand.Float64() - 1
	r := u*u + v*v
	for r == 0 || r > 1 {
		u = 2*rand.Float64() - 1
		v = 2*rand.Float64() - 1
		r = u*u + v*v
	}
	c := math.Sqrt(-2 * math.Log(r) / r)
	return u * c
}

// return random normal number
func randn(mu, std float64) float64 {
	return mu + gaussRandom()*std
}

// returns 2d array filled with random numbers
func randn2d(n int) []Point {
	res := make([]Point, n)
	for i := range res {
		for j := range res[i] {
			res[i][j] = randn(0.0, 1e-4)
		}
	}
	return res
}

// returns 2d array filled with 'val'
func fill2d(n int, val float64) []Point {
	res := make([]Point, n)
	for i := range res {
		for j := range res[i] {
			res[i][j] = val
		}
	}
	return res
}

// compute pairwise distance in all vectors in X
func xtod(x Distancer) []float64 {
	length := x.Len()
	dists := make([]float64, length*length) // allocate contiguous array
	for i := 0; i < length-1; i++ {
		for j := i + 1; j < length; j++ {
			d := x.Distance(i, j)
			dists[i*length+j] = d
			dists[j*length+i] = d
		}
	}
	return dists
}

// "constants" for positive and negative infinity
var (
	inf     = math.Inf(1)
	negInf  = math.Inf(-1)
	hTarget = math.Log(perplexity) // target entropy of distribution
)

// compute (p_{i|j} + p_{j|i})/(2n)
func d2p(D []float64, perplexity, tol float64) []float64 {
	length := int(math.Sqrt(float64(len(D))))
	pTemp := make([]float64, length*length) // temporary probability matrix
	prow := make([]float64, length)         // a temporary storage compartment
	for i := 0; i < length; i++ {
		betamin := negInf
		betamax := inf
		beta := 1.0 // initial value of precision
		const maxtries = 500
		// perform binary search to find a suitable precision beta
		// so that the entropy of the distribution is appropriate
		for num := 0; num < maxtries; num++ {
			// compute entropy and kernel row with beta precision
			psum := 0.0
			for j := 0; j < length; j++ {
				if i != j { // we dont care about diagonals
					pj := math.Exp(-D[i*length+j] * beta)
					prow[j] = pj
					psum += pj
				} else {
					prow[j] = 0.0
				}
			}
			// normalize p and compute entropy
			hHere := 0.0
			for j := 0; j < length; j++ {
				pj := prow[j] / psum
				prow[j] = pj
				if pj > 1e-7 {
					hHere -= pj * math.Log(pj)
				}
			}
			// adjust beta based on result
			if hHere > hTarget {
				// entropy was too high (distribution too diffuse)
				// so we need to increase the precision for more peaky distribution
				betamin = beta // move up the bounds
				if betamax == inf {
					beta = beta * 2
				} else {
					beta = (beta + betamax) / 2
				}
			} else {
				// converse case. make distrubtion less peaky
				betamax = beta
				if betamin == negInf {
					beta = beta / 2
				} else {
					beta = (beta + betamin) / 2
				}
			}
			// stopping conditions: too many tries or got a good precision
			if math.Abs(hHere-hTarget) < tol {
				break
			}
		}
		// copy over the final prow to P at row i
		for j := 0; j < length; j++ {
			pTemp[i*length+j] = prow[j]
		}
	} // end loop over examples i
	// symmetrize P and normalize it to sum to 1 over all ij
	probas := make([]float64, length*length)
	length2 := float64(length * 2)
	for i := 0; i < length; i++ {
		for j := 0; j < length; j++ {
			probas[i*length+j] = math.Max((pTemp[i*length+j]+pTemp[j*length+i])/length2, 1e-100)
		}
	}
	return probas
}

func sign(x float64) int {
	if x > 0 {
		return 1
	} else if x < 0 {
		return -1
	}
	return 0
}
