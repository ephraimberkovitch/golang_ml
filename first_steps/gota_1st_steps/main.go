package main

import (
	"fmt"
	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
)

func main() {
	df := dataframe.LoadRecords(
		[][]string{
			[]string{"A", "B", "C", "D"},
			[]string{"a", "4", "5.1", "true"},
			[]string{"k", "5", "7.0", "true"},
			[]string{"k", "4", "6.0", "true"},
			[]string{"a", "2", "7.1", "false"},
		},
	)
	// Change column C with a new one
	mut := df.Mutate(
		series.New([]string{"a", "b", "c", "d"}, series.String, "C"),
	)
	// Add a new column E
	mut2 := df.Mutate(
		series.New([]string{"a", "b", "c", "d"}, series.String, "E"),
	)
	fmt.Println(mut)
	fmt.Println(mut2)

	fmt.Println(df.Describe())
	fmt.Println(df.Dims())
	fmt.Println(mut2.Dims())
}
