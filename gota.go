package main

import (
	"bufio"
	"bytes"
	"fmt"
	"github.com/kniren/gota/dataframe"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"io/ioutil"
	"strconv"
	"strings"
)

type Pair struct {
	Phrase    string
	Frequency int
}

func pairsAndFilters(splitPairs []string) ([]Pair, map[string]bool) {
	var (
		pairs []Pair
		m     map[string]bool
	)
	m = make(map[string]bool)
	for _, pair := range splitPairs {
		p := strings.Split(pair, ":")
		phrase := p[0]
		m[phrase] = true
		if len(p) < 2 {
			continue
		}
		freq, err := strconv.Atoi(p[1])
		if err != nil {
			continue
		}
		pairs = append(pairs, Pair{
			Phrase:    phrase,
			Frequency: freq,
		})
	}
	return pairs, m
}

func exclude(pairs []Pair, exclusions map[string]bool) []Pair {
	var ret []Pair
	for i := range pairs {
		if !exclusions[pairs[i].Phrase] {
			ret = append(ret, pairs[i])
		}
	}
	return ret
}

func SeriesToPlotValues(df dataframe.DataFrame, col string) plotter.Values {
	rows, _ := df.Dims()
	v := make(plotter.Values, rows)
	s := df.Col(col)
	for i := 0; i < rows; i++ {
		v[i] = s.Elem(i).Float()
	}
	return v
}

func HistogramData(v plotter.Values, title string) []byte {
	// Make a plot and set its title.
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = title
	h, err := plotter.NewHist(v, 10)
	if err != nil {
		panic(err)
	}
	//h.Normalize(1) // Uncomment to normalize the area under the histogram to 1
	p.Add(h)
	w, err := p.WriterTo(5*vg.Inch, 4*vg.Inch, "jpg")
	if err != nil {
		panic(err)
	}
	var b bytes.Buffer
	writer := bufio.NewWriter(&b)
	w.WriteTo(writer)
	return b.Bytes()
	// p.Save(5*vg.Inch, 4*vg.Inch, title + ".png")
}

func SaveHistogram(v plotter.Values, title string) {
	// Make a plot and set its title.
	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = title
	h, err := plotter.NewHist(v, 10)
	if err != nil {
		panic(err)
	}
	h.Normalize(1) // Uncomment to normalize the area under the histogram to 1
	p.Add(h)
	p.Save(5*vg.Inch, 4*vg.Inch, "./output/"+title+".png")
}

func main() {
	const kitchenReviews = "./datasets/words/kitchen"
	positives, err := ioutil.ReadFile(kitchenReviews + "/positive.review")
	negatives, err2 := ioutil.ReadFile(kitchenReviews + "/negative.review")
	if err != nil || err2 != nil {
		fmt.Println("Error(s)", err, err2)
	}

	pairsPositive := strings.Fields(string(positives))
	pairsNegative := strings.Fields(string(negatives))

	parsedPositives, posPhrases := pairsAndFilters(pairsPositive)
	parsedNegatives, negPhrases := pairsAndFilters(pairsNegative)
	parsedPositives = exclude(parsedPositives, negPhrases)
	parsedNegatives = exclude(parsedNegatives, posPhrases)

	dfPos := dataframe.LoadStructs(parsedPositives)
	dfNeg := dataframe.LoadStructs(parsedNegatives)

	dfPos = dfPos.Arrange(dataframe.RevSort("Frequency"))
	dfNeg = dfNeg.Arrange(dataframe.RevSort("Frequency"))

	fmt.Println(dfPos)
	fmt.Println(dfNeg)

	path := "./datasets/bmi/500_Person_Gender_Height_Weight_Index.csv"
	b, err := ioutil.ReadFile(path)
	if err != nil {
		fmt.Println("Error!", err)
	}
	df := dataframe.ReadCSV(bytes.NewReader(b))
	fmt.Println(df)

	fmt.Println("Minimum", df.Col("Height").Min())
	fmt.Println("Maximum", df.Col("Height").Max())
	fmt.Println("Mean", df.Col("Height").Mean())
	fmt.Println("Median", df.Col("Height").Quantile(0.5))

	fmt.Println(df.Describe())

	jpeg := HistogramData(SeriesToPlotValues(df, "Height"), "Height Histogram")
	_ = ioutil.WriteFile("./output/1.jpeg", jpeg, 0644)
	SaveHistogram(SeriesToPlotValues(df, "Height"), "Height Histogram")
}
