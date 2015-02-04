package tsne4go

// Distancer describes a collection of points from which a distance can be computed
type Distancer interface {
	Len() int                  // Length of the collection
	Distance(i, j int) float64 // Distance between items i and j of the collection
}

// A Distancer implemented for vectors of float64
type VectorDistancer [][]float64

func (vd VectorDistancer) Len() int { return len(vd) }

func (vd VectorDistancer) Distance(i, j int) float64 {
	vi := vd[i]
	vj := vd[j]
	dist := 0.0
	for k, vik := range vi {
		vjk := vj[k]
		dist += (vik - vjk) * (vik - vjk)
	}
	return dist
}
