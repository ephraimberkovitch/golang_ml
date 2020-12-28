package main

import (
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"image/color"
	"log"
	"os"
)

func main() {

	pts := make(plotter.XYs, 16)
	for index, _ := range pts {
		pts[index].X = float64(index + 1)
		pts[index].Y = float64(index + 1)
	}

	data := pts

	p, err := plot.New()
	if err != nil {
		log.Panic(err)
	}
	p.Title.Text = "Linear Plot"
	p.X.Label.Text = "Indices"
	p.Y.Label.Text = "Squares"
	p.Add(plotter.NewGrid())

	line, points, err := plotter.NewLinePoints(data)
	if err != nil {
		log.Panic(err)
	}
	line.Color = color.RGBA{B: 255, A: 255}

	p.Add(line, points)

	pts = make(plotter.XYs, 16)
	for index, _ := range pts {
		pts[index].X = float64(index + 1)
		pts[index].Y = float64((index + 1) * (index + 1))
	}

	data = pts

	line, points, err = plotter.NewLinePoints(data)
	if err != nil {
		log.Panic(err)
	}
	line.Color = color.RGBA{R: 255, A: 255}

	p.Add(line, points)

	path, err := os.Getwd()
	if err != nil {
		log.Panic(err)
	}
	err = p.Save(10*vg.Centimeter, 5*vg.Centimeter, path+"/plot_example.png")
	if err != nil {
		log.Panic(err)
	}
}
