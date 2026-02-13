package main

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

type Config struct {
	APIKeyID  string
	SecretKey string
	Paper     bool
	BaseURL   string
	DataURL   string
}

// LoadConfig loads credentials from .env file next to the binary,
// then falls back to environment variables.
func LoadConfig() (*Config, error) {
	// Try loading .env from the directory of the executable, then CWD
	for _, dir := range []string{exeDir(), "."} {
		envPath := filepath.Join(dir, ".env")
		if err := loadDotEnv(envPath); err == nil {
			break
		}
	}

	cfg := &Config{
		APIKeyID:  os.Getenv("APCA_API_KEY_ID"),
		SecretKey: os.Getenv("APCA_API_SECRET_KEY"),
		BaseURL:   os.Getenv("APCA_API_BASE_URL"),
		DataURL:   os.Getenv("APCA_API_DATA_URL"),
	}

	paperStr := os.Getenv("PAPER")
	cfg.Paper = paperStr == "" || paperStr == "true" || paperStr == "1"

	if cfg.BaseURL == "" {
		if cfg.Paper {
			cfg.BaseURL = "https://paper-api.alpaca.markets"
		} else {
			cfg.BaseURL = "https://api.alpaca.markets"
		}
	}
	if cfg.DataURL == "" {
		cfg.DataURL = "https://data.alpaca.markets"
	}

	if cfg.APIKeyID == "" || cfg.SecretKey == "" {
		return nil, fmt.Errorf("missing APCA_API_KEY_ID or APCA_API_SECRET_KEY (set in .env or environment)")
	}

	return cfg, nil
}

func exeDir() string {
	exe, err := os.Executable()
	if err != nil {
		return "."
	}
	return filepath.Dir(exe)
}

// loadDotEnv reads a simple KEY=VALUE .env file and sets env vars
// (only if not already set in the environment).
func loadDotEnv(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(parts[1])
		// Don't override existing env vars
		if os.Getenv(key) == "" {
			os.Setenv(key, val)
		}
	}
	return scanner.Err()
}
