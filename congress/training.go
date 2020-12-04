package main

import (
	"encoding/gob"
	"fmt"
	"image/color"
	"log"
	"math"
	"os"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	g := gorgonia.NewGraph()
	x, y := getXYMat()
	plotData(x, y)

	xT := tensor.FromMat64(mat.DenseCopyOf(x))
	yT := tensor.FromMat64(mat.DenseCopyOf(y))

	s := yT.Shape()
	yT.Reshape(s[0])

	X := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))
	Y := gorgonia.NodeFromAny(g, yT, gorgonia.WithName("y"))
	theta := gorgonia.NewVector(
		g,
		gorgonia.Float64,
		gorgonia.WithName("theta"),
		gorgonia.WithShape(xT.Shape()[1]),
		gorgonia.WithInit(gorgonia.Gaussian(0, 1)))

	pred := must(gorgonia.Mul(X, theta))

	// Gorgonia might delete values from nodes so we are going to save it
	// and print it out later
	var predicted gorgonia.Value
	gorgonia.Read(pred, &predicted)

	squaredError := must(gorgonia.Square(must(gorgonia.Sub(pred, Y))))
	cost := must(gorgonia.Mean(squaredError))

	if _, err := gorgonia.Grad(cost, theta); err != nil {
		log.Fatalf("Failed to backpropagate: %v", err)
	}

	machine := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(theta))
	defer machine.Close()

	model := []gorgonia.ValueGrad{theta}
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(0.001))

	fa := mat.Formatted(getThetaNormal(x, y), mat.Prefix("   "), mat.Squeeze())

	fmt.Printf("ϴ: %v\n", fa)
	iter := 10000
	var err error
	for i := 0; i < iter; i++ {
		if err = machine.RunAll(); err != nil {
			fmt.Printf("Error during iteration: %v: %v\n", i, err)
			break
		}

		if err = solver.Step(model); err != nil {
			log.Fatal(err)
		}
		fmt.Printf("theta: %2.2f  Iter: %v Cost: %2.3f Accuracy: %2.2f \r",
			theta.Value(),
			i,
			cost.Value(),
			accuracy(predicted.Data().([]float64), Y.Value().Data().([]float64)))

		machine.Reset() // Reset is necessary in a loop like this
	}
	fmt.Println("")
	err = save(theta.Value())
	if err != nil {
		log.Fatal(err)
	}

}

func accuracy(prediction, y []float64) float64 {
	var ok float64
	for i := 0; i < len(prediction); i++ {
		if math.Round(prediction[i]-y[i]) == 0 {
			ok += 1.0
		}
	}
	return ok / float64(len(y))
}

func getXYMat() (*matrix, *matrix) {
	f, err := os.Open("/Users/ephraimb/berkotech/golang_ml/congress/congressional_voting_dataset_raw.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	df := dataframe.ReadCSV(f)
	df.Describe()

	mapVotes := map[string]float64{
		"y": 1.0,
		"n": 0.0,
		"?": 0.5,
	}
	voteToValue := func(s series.Series) series.Series {
		records := s.Records()
		floats := make([]float64, len(records))
		for i, r := range records {
			floats[i] = mapVotes[r]
		}
		return series.Floats(floats)
	}

	mapParties := map[string]float64{
		"republican": 1.0,
		"democrat":   2.0,
	}
	partyToValue := func(s series.Series) series.Series {
		records := s.Records()
		floats := make([]float64, len(records))
		for i, r := range records {
			floats[i] = mapParties[r]
		}
		return series.Floats(floats)
	}

	xDF := df.Drop("political_party").Capply(voteToValue)
	yDF := df.Select("political_party").Capply(partyToValue)
	numRows, _ := xDF.Dims()
	xDF = xDF.Mutate(series.New(one(numRows), series.Float, "bias"))
	fmt.Println(xDF.Describe())
	fmt.Println(yDF.Describe())

	return &matrix{xDF}, &matrix{yDF}
}

func getThetaNormal(x, y *matrix) *mat.Dense {
	xt := mat.DenseCopyOf(x).T()
	var xtx mat.Dense
	xtx.Mul(xt, x)
	var invxtx mat.Dense
	invxtx.Inverse(&xtx)
	var xty mat.Dense
	xty.Mul(xt, y)
	var output mat.Dense
	output.Mul(&invxtx, &xty)

	return &output
}

type matrix struct {
	dataframe.DataFrame
}

func (m matrix) At(i, j int) float64 {
	return m.Elem(i, j).Float()
}

func (m matrix) T() mat.Matrix {
	return mat.Transpose{Matrix: m}
}

func must(n *gorgonia.Node, err error) *gorgonia.Node {
	if err != nil {
		panic(err)
	}
	return n
}

func one(size int) []float64 {
	one := make([]float64, size)
	for i := 0; i < size; i++ {
		one[i] = 1.0
	}
	return one
}

func save(value gorgonia.Value) error {
	f, err := os.Create("/Users/ephraimb/berkotech/golang_ml/congress/theta.bin")
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	err = enc.Encode(value)
	if err != nil {
		return err
	}
	return nil
}

func plotData(x *matrix, y *matrix) {
	ptsDems := make(plotter.XYs, 16)
	ptsReps := make(plotter.XYs, 16)

	for i, _ := range ptsDems {
		ptsDems[i].X = float64(i + 1)
		ptsReps[i].X = float64(i + 1)

		votesDems := 0.0
		votesReps := 0.0

		for j := 0; j < x.DataFrame.Nrow(); j++ {
			if y.DataFrame.Elem(j, 0).Float() == 1 {
				votesReps += x.DataFrame.Elem(j, i).Float()
			} else {
				votesDems += x.DataFrame.Elem(j, i).Float()
			}
		}
		ptsDems[i].Y = votesDems
		ptsReps[i].Y = votesReps
	}

	data := ptsDems

	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}
	p.Title.Text = "Linear Plot"
	p.X.Label.Text = "Laws"
	p.Y.Label.Text = "Votes"
	p.Add(plotter.NewGrid())

	line, points, err := plotter.NewLinePoints(data)
	if err != nil {
		log.Panic(err)
	}
	line.Color = color.RGBA{B: 255, A: 255}

	p.Add(line, points)

	data = ptsReps

	line, points, err = plotter.NewLinePoints(data)
	if err != nil {
		log.Panic(err)
	}
	line.Color = color.RGBA{R: 255, A: 255}

	p.Add(line, points)

	err = p.Save(10*vg.Centimeter, 5*vg.Centimeter, "/Users/ephraimb/berkotech/golang_ml/congress/votes.png")
	if err != nil {
		log.Panic(err)
	}
}
