// TODO:
// Package ada implements

package ada

import (
	"encoding/json"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"sync"

	"github.com/james-bowman/nlp"
	"github.com/james-bowman/nlp/measures/pairwise"
	"gonum.org/v1/gonum/mat"
)

// Ada struct holding data shared across all Sessions
// created from this instance.
type Ada struct {
	mu       sync.RWMutex
	Dataset  Dataset
	Fallback []string
}

// Single session used for chatting with Ada.
type Session struct {
	ada          *Ada
	lastResponse *string
}

// Type MessagePair implements the most basic
// storage for 'input' and 'output' messages
type MessagePair struct {
	Input    string `json:"input"`
	Response string `json:"response"`
}

// Dataset holding all 'input' and 'output' messages
// for Ada to use
type Dataset struct {
	Pairs []MessagePair `json:"pairs"`
}

// Pre-defined dataset for basic responses
var defaultDataset = []MessagePair{
	{Input: "Hi!", Response: "Hello! How are you?"},
	{Input: "Hello!", Response: "Hi there!"},
	{Input: "How are you?", Response: "I'm fine, thanks for asking!"},
	{Input: "What's your name?", Response: "I'm Ada, your lovely chatbot."},
	{Input: "What do you like?", Response: "I like chatting with you!"},
}

// Pre-defined fallback responses if no response could be found.
var fallbackDataset = []string{
	"Interesting!",
	"Tell me more.",
	"Hmm, okay.",
	"Why do you say that?",
	"Go on.",
}

// Creates a new Ada instance with default parameters.
func NewAda() *Ada {
	return &Ada{
		Dataset:  Dataset{Pairs: defaultDataset},
		Fallback: fallbackDataset,
	}
}

// Creates a new session for chatting to Ada.
func (ada *Ada) NewSession() *Session {
	return &Session{ada: ada}
}

// Loads dataset from .json dataset into Ada instance.
//
// This operation overwrites any existing dataset in instance.
func (ada *Ada) LoadDataset(path string) *Ada {
	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		// TODO: Proper error handling
		panic(err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		// TODO: Proper error handling
		panic(err)
	}

	var ds Dataset
	if err := json.Unmarshal(data, &ds); err != nil {
		// TODO: Proper error handling
		panic(err)
	}

	ada.Dataset = ds
	return ada
}

// Saves dataset to .json file from current Ada instance.
//
// This operation overwrites any existing data in file.
func (ada *Ada) SaveDataset(path string) {
	data, err := json.MarshalIndent(ada.Dataset, "", "  ")
	if err != nil {
		// TODO: Proper error handling
		panic(err)
	}
	os.WriteFile(path, data, 0644)
}

type vectorizerSnapshot struct {
	pipeline *nlp.Pipeline
	vectors  mat.Matrix
	pairs    []MessagePair
}

func (ada *Ada) buildVectorizer() *vectorizerSnapshot {
	ada.mu.RLock()
	pairs := make([]MessagePair, len(ada.Dataset.Pairs))
	copy(pairs, ada.Dataset.Pairs)
	ada.mu.RUnlock()

	inputs := make([]string, len(pairs))
	for i, p := range pairs {
		inputs[i] = p.Input
	}

	vectorizer := nlp.NewPipeline(
		nlp.NewCountVectoriser(),
		nlp.NewTfidfTransformer(),
	)

	inputVectors, err := vectorizer.FitTransform(inputs...)
	if err != nil {
		// TODO: Proper error handling
		panic(err)
	}

	return &vectorizerSnapshot{pipeline: vectorizer, vectors: inputVectors, pairs: pairs}
}

// Returns best response to input string found in database along with score [0..1] (roughly)
//
// topN - how many of the top responses to consider for response
//
// Lower score - less confident in response, higher score - more confident in response
// Requires at least 0.2 score to return response from database, otherwise, returns nil
func (ada *Ada) findBestResponse(snap *vectorizerSnapshot, input string, topN int) (*string, float64) {
	userMatrix, err := snap.pipeline.Transform(input)
	if err != nil {
		// TODO: Proper error handling
		panic(err)
	}

	userVec := columnVector(userMatrix, 0)

	_, numDocs := snap.vectors.Dims()

	sims := make([]float64, numDocs)
	for i := range numDocs {
		s := pairwise.CosineSimilarity(userVec, columnVector(snap.vectors, i))
		if math.IsNaN(s) {
			s = 0
		}
		sims[i] = s
	}

	// Sort indices by similarity, descending
	indices := make([]int, numDocs)
	for i := range indices {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		return sims[indices[i]] > sims[indices[j]]
	})

	if topN > len(indices) {
		topN = len(indices)
	}
	topIndices := indices[:topN]

	topScores := make([]float64, topN)
	maxScore := sims[topIndices[0]]
	total := 0.0
	for i, idx := range topIndices {
		topScores[i] = sims[idx]
		total += sims[idx]
	}

	if maxScore < 0.2 {
		return nil, maxScore
	}

	probs := make([]float64, topN)
	if total == 0 {
		for i := range probs {
			probs[i] = 1.0 / float64(topN)
		}
	} else {
		for i, s := range topScores {
			probs[i] = s / total
		}
	}

	chosenLocal := weightedChoice(probs)
	chosenIdx := topIndices[chosenLocal]
	score := sims[chosenIdx]

	if score < 0.2 {
		return nil, score
	}

	response := snap.pairs[chosenIdx].Response
	return &response, score
}

func weightedChoice(probs []float64) int {
	r := rand.Float64()
	cumulative := 0.0
	for i, p := range probs {
		cumulative += p
		if r < cumulative {
			return i
		}
	}
	return len(probs) - 1
}

func columnVector(m mat.Matrix, col int) *mat.VecDense {
	rows, _ := m.Dims()
	data := make([]float64, rows)
	for r := range rows {
		data[r] = m.At(r, col)
	}
	return mat.NewVecDense(rows, data)
}

// Returns best possible response to input text.
// If no "good enough" response could be found, returns a random
// string from the fallback dataset.
//
// This function also trains the database with the new input.
//
// 2nd return parameter is response confidence.
// Lower score - less confident in response, higher score - more confident in response.
// This value ranges from [0..1] (approximately)
// Values below 0.2 are treated as not confident enough and will return fallback strings.
func (s *Session) GetResponse(input string) (string, float64) {
	input = strings.TrimSpace(input)
	// No input
	if input == "" {
		return "", 0.0
	}

	// Train
	if s.lastResponse != nil {
		newPair := MessagePair{Input: *s.lastResponse, Response: input}
		s.ada.mu.Lock()
		s.ada.Dataset.Pairs = append(s.ada.Dataset.Pairs, newPair)
		s.ada.mu.Unlock()
	}

	snap := s.ada.buildVectorizer()
	reply, score := s.ada.findBestResponse(snap, input, 3)

	// No best response found, using fallback
	if reply == nil {
		fallback := s.ada.Fallback[rand.Intn(len(s.ada.Fallback))]
		reply = &fallback
	}

	s.lastResponse = reply
	return *reply, score
}
