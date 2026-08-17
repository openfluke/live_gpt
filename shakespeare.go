package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"time"
	"unicode/utf8"
)

// Same tinyshakespeare file nanoGPT shakespeare_char uses. GitHub raw 429s
// Go's default User-Agent, so we send one and fall through CDN mirrors.
var shakespeareURLs = []string{
	"https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt",
	"https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
	"https://github.com/karpathy/char-rnn/raw/master/data/tinyshakespeare/input.txt",
}

// Corpus is a character-level token stream (nanoGPT shakespeare_char style).
type Corpus struct {
	IDs   []int
	Chars []rune // id → rune
	Index map[rune]int
}

func (c *Corpus) Vocab() int {
	if c == nil {
		return 0
	}
	return len(c.Chars)
}

func (c *Corpus) Encode(s string) []int {
	out := make([]int, 0, utf8.RuneCountInString(s))
	for _, r := range s {
		id, ok := c.Index[r]
		if !ok {
			continue
		}
		out = append(out, id)
	}
	return out
}

// LoadShakespeare downloads tinyshakespeare into dir (once) and builds a char vocab.
func LoadShakespeare(dir string) (*Corpus, error) {
	if dir == "" {
		dir = "data"
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "tinyshakespeare.txt")
	if _, err := os.Stat(path); err != nil {
		if err := downloadShakespeare(path); err != nil {
			return nil, err
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return CorpusFromString(string(raw)), nil
}

// CorpusFromString builds a stable sorted char vocab (tests / tiny probes).
func CorpusFromString(s string) *Corpus {
	seen := map[rune]bool{}
	var ids []int
	runes := []rune(s)
	for _, r := range runes {
		seen[r] = true
	}
	chars := make([]rune, 0, len(seen))
	for r := range seen {
		chars = append(chars, r)
	}
	sort.Slice(chars, func(i, j int) bool { return chars[i] < chars[j] })
	index := make(map[rune]int, len(chars))
	for i, r := range chars {
		index[r] = i
	}
	ids = make([]int, len(runes))
	for i, r := range runes {
		ids[i] = index[r]
	}
	return &Corpus{IDs: ids, Chars: chars, Index: index}
}

func downloadShakespeare(path string) error {
	var last error
	for _, url := range shakespeareURLs {
		err := download(path, url)
		if err == nil {
			return nil
		}
		last = err
		time.Sleep(400 * time.Millisecond)
	}
	return last
}

func download(path, url string) error {
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", "live_gpt (Welvet tide host; +https://github.com/openfluke)")
	req.Header.Set("Accept", "text/plain,*/*")
	client := &http.Client{Timeout: 45 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("shakespeare: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("shakespeare: GET %s → %s", url, resp.Status)
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, err = io.Copy(f, resp.Body)
	cerr := f.Close()
	if err != nil {
		_ = os.Remove(tmp)
		return err
	}
	if cerr != nil {
		_ = os.Remove(tmp)
		return cerr
	}
	return os.Rename(tmp, path)
}
