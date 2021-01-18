package main

import (
	"bufio"
	"bytes"
	"encoding/gob"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"os"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// https://www.kaggle.com/amarpandey/implementing-linear-regression-on-iris-dataset/notebook
//
func main() {
	g := gorgonia.NewGraph()
	x, y := getXYMat()
	path, err := os.Getwd()
	if err != nil {
		log.Panic(err)
	}

	plotData(x.Col("sepal_length").Float(), x.Col("sepal_width").Float(), y.Col("species").Float(), path+"/sepal.png", "sepal")
	plotData(x.Col("petal_length").Float(), x.Col("petal_width").Float(), y.Col("species").Float(), path+"/petal.png", "petal")

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

	if _, err = gorgonia.Grad(cost, theta); err != nil {
		log.Fatalf("Failed to backpropagate: %v", err)
	}

	machine := gorgonia.NewTapeMachine(g, gorgonia.BindDualValues(theta))
	defer machine.Close()

	model := []gorgonia.ValueGrad{theta}
	solver := gorgonia.NewVanillaSolver(gorgonia.WithLearnRate(0.001))

	fa := mat.Formatted(getThetaNormal(x, y), mat.Prefix("   "), mat.Squeeze())
	fmt.Printf("ϴ: %v\n", fa)

	iter := 100000
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
	path, err := os.Getwd()
	if err != nil {
		log.Panic(err)
	}
	f, err := os.Open(path + "/iris.csv")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	df := dataframe.ReadCSV(f)
	df.Describe()

	toValue := func(s series.Series) series.Series {
		records := s.Records()
		floats := make([]float64, len(records))
		m := map[string]int{}
		for i, r := range records {
			if _, ok := m[r]; !ok {
				m[r] = len(m) + 1
			}
			floats[i] = float64(m[r])
		}
		return series.Floats(floats)
	}

	xDF := df.Drop("species")
	yDF := df.Select("species").Capply(toValue)
	// plotData(df.Col("sepal_length").Float(), df.Col("sepal_width").Float(),yDF.Col("species").Float())
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
	path, err := os.Getwd()
	if err != nil {
		log.Panic(err)
	}
	f, err := os.Create(path + "/theta.bin")
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

func plotData(x []float64, y []float64, a []float64, fileName string, topic string) []byte {
	p, err := plot.New()
	if err != nil {
		log.Fatal(err)
	}

	p.Title.Text = topic + " length & width"
	p.X.Label.Text = "length"
	p.Y.Label.Text = "width"
	p.Add(plotter.NewGrid())

	for k := 1; k <= 3; k++ {
		data0 := make(plotter.XYs, 0)
		for i := 0; i < len(a)-1; i++ {
			if k != int(a[i]) {
				continue
			}
			x1 := x[i]
			y1 := y[i]
			data0 = append(data0, plotter.XY{X: x1, Y: y1})
		}
		data, err := plotter.NewScatter(data0)
		if err != nil {
			log.Fatal(err)
		}
		data.GlyphStyle.Color = plotutil.Color(k - 1)
		data.Shape = &draw.PyramidGlyph{}
		p.Add(data)
		p.Legend.Add(fmt.Sprint(k), data)
	}

	w, err := p.WriterTo(4*vg.Inch, 4*vg.Inch, "png")
	if err != nil {
		panic(err)
	}
	var b bytes.Buffer
	writer := bufio.NewWriter(&b)
	w.WriteTo(writer)
	ioutil.WriteFile(fileName, b.Bytes(), 0644)
	return b.Bytes()
}
