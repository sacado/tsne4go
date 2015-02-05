package tsne4go

import "math"

type TSne struct {
	perplexity float64
	dim        int
	epsilon    float64
	iter       int
	length     int
	probas     []float64
	Solution   [][]float64
	gains      [][]float64
	ystep      [][]float64
}

const (
	defaultPerplexity = 30
	defaultDim        = 2
	defaultEpsilon    = 10
)

// NewTSne takes a set of Distancer instances
// and creates matrix P from them using gaussian kernel
func NewTSne(x Distancer) *TSne {
	dists := xtod(x) // convert x to distances using gaussian kernel
	tsne := &TSne{
		defaultPerplexity,    // perplexity
		defaultDim,           // dim
		defaultEpsilon,       // epsilon
		0,                    // iters
		x.Len(),              //length
		d2p(dists, 30, 1e-4), // probas
		nil,                  // Solution
		nil,                  // gains
		nil,                  // ystep
	}
	tsne.initSolution() // refresh this
	return tsne
}

// (re)initializes the solution to random
func (tsne *TSne) initSolution() {
	// generate random solution to t-SNE
	tsne.Solution = randn2d(tsne.length, tsne.dim)  // the solution
	tsne.gains = fill2d(tsne.length, tsne.dim, 1.0) // step gains to accelerate progress in unchanging directions
	tsne.ystep = fill2d(tsne.length, tsne.dim, 0.0) // momentum accumulator
	tsne.iter = 0
}

// perform a single step of optimization to improve the embedding
func (tsne *TSne) Step() float64 {
	tsne.iter++
	length := tsne.length
	cost, grad := tsne.costGrad(tsne.Solution) // evaluate gradient
	// perform gradient step
	ymean := make([]float64, tsne.dim)
	for i := 0; i < length; i++ {
		for d := 0; d < tsne.dim; d++ {
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
			var momval float64
			if tsne.iter < 250 {
				momval = 0.5
			} else {
				momval = 0.8
			}
			newsid := momval*sid - tsne.epsilon*tsne.gains[i][d]*grad[i][d]
			tsne.ystep[i][d] = newsid // remember the step we took
			// step!
			tsne.Solution[i][d] += newsid
			ymean[d] += tsne.Solution[i][d] // accumulate mean so that we can center later
		}
	}
	// reproject Y to be zero mean
	for i := 0; i < length; i++ {
		for d := 0; d < tsne.dim; d++ {
			tsne.Solution[i][d] -= ymean[d] / float64(length)
		}
	}
	return cost
}

// return cost and gradient, given an arrangement
func (tsne *TSne) costGrad(Y [][]float64) (cost float64, grad [][]float64) {
	length := tsne.length
	dim := tsne.dim // dim of output space
	P := tsne.probas
	var pmul float64
	if tsne.iter < 100 { // trick that helps with local optima
		pmul = 4.0
	} else {
		pmul = 1.0
	}
	// compute current Q distribution, unnormalized first
	Qu := make([]float64, length*length)
	qsum := 0.0
	for i := 0; i < length-1; i++ {
		for j := i + 1; j < length; j++ {
			dsum := 0.0
			for d := 0; d < dim; d++ {
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
	squareLength := length * length
	Q := make([]float64, squareLength)
	for q := range Q {
		Q[q] = math.Max(Qu[q]/qsum, 1e-100)
	}
	cost = 0.0
	grad = [][]float64{}
	for i := 0; i < length; i++ {
		gsum := make([]float64, dim) // init grad for point i
		for j := 0; j < length; j++ {
			cost += -P[i*length+j] * math.Log(Q[i*length+j]) // accumulate cost (the non-constant portion at least...)
			premult := 4 * (pmul*P[i*length+j] - Q[i*length+j]) * Qu[i*length+j]
			for d := 0; d < dim; d++ {
				gsum[d] += premult * (Y[i][d] - Y[j][d])
			}
		}
		grad = append(grad, gsum)
	}

	return cost, grad
}

// Normalize makes all values from the solution in the interval [0; 1]
func (tsne *TSne) NormalizeSolution() {
	mins := make([]float64, tsne.dim)
	maxs := make([]float64, tsne.dim)
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
