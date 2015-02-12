package tsne4go

import (
	"math"
	"sync"
)

const (
	perplexity = 30
	nbDims     = 2
	epsilon    = 10
)

type Point [nbDims]float64

type TSne struct {
	iter int
	// All subsequent vector should have 'length' elements
	length   int
	probas   []float64
	Solution []Point
	gains    []Point
	ystep    []Point
	// Meta-information about each point, if needed
	// It is useful to associate, for instance, a label with each point
	// The algorithm dosen't take this information into consideration
	// It can be anything, even nil if the user has no need for it
	Meta []interface{}
}

// New takes a set of Distancer instances
// and creates matrix P from them using gaussian kernel
// Meta-information is provided here
// It is under the programmer's responsibility :
// it can be nil if no meta information is needed, or anything else
func New(x Distancer, meta []interface{}) *TSne {
	dists := xtod(x) // convert x to distances using gaussian kernel
	length := x.Len()
	tsne := &TSne{
		0,      // iters
		length, // length
		d2p(dists, perplexity, 1e-7), // probas ( was 1e-4)
		randn2d(length),              // Solution
		fill2d(length, 1.0),          // gains
		make([]Point, length),        // ystep
		meta, // Meta
	}
	return tsne
}

// perform a single step of optimization to improve the embedding
func (tsne *TSne) Step() float64 {
	tsne.iter++
	length := tsne.length
	cost, grad := tsne.costGrad(tsne.Solution) // evaluate gradient
	var ymean [nbDims]float64
	var wg sync.WaitGroup
	// perform gradient step
	for d := 0; d < nbDims; d++ {
		go func(d int) {
			wg.Add(1)
			defer wg.Done()
			for i := 0; i < length; i++ {
				gid := grad[i][d]
				sid := tsne.ystep[i][d]
				gainid := tsne.gains[i][d]
				// compute gain update
				if sign(gid) == sign(sid) {
					tsne.gains[i][d] = gainid * 0.8
				} else {
					tsne.gains[i][d] = gainid + 0.2
				}
				// compute momentum step direction
				momval := 0.8
				if tsne.iter < 250 {
					momval = 0.5
				}
				newsid := momval*sid - epsilon*tsne.gains[i][d]*grad[i][d]
				tsne.ystep[i][d] = newsid       // remember the step we took
				tsne.Solution[i][d] += newsid   // step
				ymean[d] += tsne.Solution[i][d] // accumulate mean so that we can center later
			}
		}(d)
	}
	wg.Wait()
	// reproject Y to be zero mean
	for d, mean := range ymean {
		go func(d int) {
			for i := 0; i < length; i++ {
				tsne.Solution[i][d] -= mean / float64(length)
			}
		}(d)
	}
	return cost
}

// return cost and gradient, given an arrangement
func (tsne *TSne) costGrad(Y []Point) (cost float64, grad []Point) {
	length := tsne.length
	P := tsne.probas
	pmul := 1.0
	if tsne.iter < 100 { // trick that helps with local optima
		pmul = 4.0
	}
	// compute current Q distribution, unnormalized first
	squareLength := length * length
	Qu := make([]float64, squareLength)
	qsum := 0.0
	for i := 0; i < length-1; i++ {
		for j := i + 1; j < length; j++ {
			dsum := 0.0
			for d := 0; d < nbDims; d++ {
				dhere := Y[i][d] - Y[j][d]
				dsum += dhere * dhere
			}
			qu := 1.0 / (1.0 + dsum) // Student t-distribution
			Qu[i*length+j] = qu
			Qu[j*length+i] = qu
			qsum += 2 * qu
		}
	}
	// normalize Q distribution to sum to 1
	Q := make([]float64, squareLength)
	for q := range Q {
		Q[q] = math.Max(Qu[q]/qsum, 1e-100)
	}
	cost = 0.0
	grad = make([]Point, length)
	for i := 0; i < length; i++ {
		gsum := &grad[i]
		for j := 0; j < length; j++ {
			// accumulate cost (the non-constant portion at least...)
			idx := i*length + j
			cost += -P[idx] * math.Log(Q[idx])
			premult := 4 * (pmul*P[idx] - Q[idx]) * Qu[idx]
			for d := 0; d < nbDims; d++ {
				gsum[d] += premult * (Y[i][d] - Y[j][d])
			}
		}
	}

	return cost, grad
}

// Normalize makes all values from the solution in the interval [0; 1]
func (tsne *TSne) NormalizeSolution() {
	var mins [nbDims]float64
	var maxs [nbDims]float64
	for i, pt := range tsne.Solution {
		for j, val := range pt {
			if i == 0 || val < mins[j] {
				mins[j] = val
			}
			if i == 0 || val > maxs[j] {
				maxs[j] = val
			}
		}
	}
	for i, pt := range tsne.Solution {
		for j, val := range pt {
			tsne.Solution[i][j] = (val - mins[j]) / (maxs[j] - mins[j])
		}
	}
}
