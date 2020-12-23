package main

import (
	"bufio"
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

func main() {
	f, err := os.Open("/Users/ephraimb/berkotech/golang_ml/congress/theta.bin")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	var thetaT *tensor.Dense
	err = dec.Decode(&thetaT)
	if err != nil {
		log.Fatal(err)
	}
	g := gorgonia.NewGraph()
	theta := gorgonia.NodeFromAny(g, thetaT, gorgonia.WithName("theta"))
	values := make([]float64, 17)
	xT := tensor.New(tensor.WithBacking(values))
	x := gorgonia.NodeFromAny(g, xT, gorgonia.WithName("x"))
	y, err := gorgonia.Mul(x, theta)
	machine := gorgonia.NewTapeMachine(g)
	defer machine.Close()
	values[16] = 11
	for {
		values[0] = getInput("handicapped_infants")
		values[1] = getInput("water_project_cost_sharing")
		values[2] = getInput("adoption_of_the_budget_resolution")
		values[3] = getInput("physician_fee_freeze")
		values[4] = getInput("el_salvador_aid")
		values[5] = getInput("religious_groups_in_schools")
		values[6] = getInput("anti_satellite_test_ban")
		values[7] = getInput("aid_to_nicaraguan_contras")
		values[8] = getInput("mx_missile")
		values[9] = getInput("immigration")
		values[10] = getInput("synfuels_corporation_cutback")
		values[11] = getInput("education_spending")
		values[12] = getInput("superfund_right_to_sue")
		values[13] = getInput("crime")
		values[14] = getInput("duty_free_exports")
		values[15] = getInput("export_administration_act_south_africa")

		if err = machine.RunAll(); err != nil {
			log.Fatal(err)
		}
		val := math.Round(y.Value().Data().(float64) - 9)
		switch val {
		case 1:
			fmt.Println("You are a probably a republican")
		case 2:
			fmt.Println("You are a probably a democrat")
		}
		machine.Reset()
	}
}

func getInput(s string) float64 {
	reader := bufio.NewReader(os.Stdin)
	fmt.Printf("%v: ", s)
	text, _ := reader.ReadString('\n')
	text = strings.TrimSpace(text)

	input, err := strconv.ParseFloat(text, 64)
	if err != nil {
		log.Fatal(err)
	}
	return input
}
