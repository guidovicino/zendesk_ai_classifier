# Zendesk Ticket Processing Tools

 Python tools for processing and generating Zendesk-style support tickets with AI classification using Ollama.

## ✨ Vibe Coded with Claude Code

> ⚠️ **Disclaimer:** I'm not a developer and haven't had the chance to test this properly.  
> Still, it might be useful to you too! 🙌

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure Ollama is running:
```bash
ollama serve
```

3. Pull required model:
```bash
ollama pull mistral:7b
```

## Usage

### Ticket Classification

Basic usage with heuristics only:
```bash
python3 zendesk_ai_classifier.py tickets.json --out results.csv
```

With Ollama fallback for unclassified tickets:
```bash
python3 zendesk_ai_classifier.py tickets.json --out results.csv --ollama --model mistral:7b
```

Force Ollama for all tickets:
```bash
python3 zendesk_ai_classifier.py tickets.json --out results.csv --ollama-for-all --model mistral:7b
```

With custom configuration:
```bash
python3 zendesk_ai_classifier.py tickets.json --config config.json --verbose --batch-size 20
```

### Synthetic Data Generation

Generate 100 synthetic tickets:
```bash
python3 zendesk_synthesizer.py example_ticket.json --n 100 --out synthetic_tickets.json
```

With custom categories and model:
```bash
python3 zendesk_synthesizer.py example_ticket.json --n 50 --categories "Login,Billing,Bug" --model llama2:7b --verbose
```

## Configuration

Create a `config.json` file for persistent configuration:

```json
{
  "ollama": {
    "host": "http://localhost:11434",
    "model": "mistral:7b",
    "timeout": 90,
    "max_retries": 3,
    "backoff_factor": 1.5
  },
  "classifier": {
    "confidence_threshold": 0.7,
    "text_truncate_length": 220,
    "batch_size": 1,
    "enable_caching": true,
    "cache_size": 128
  },
  "synthesizer": {
    "default_num_tickets": 50,
    "max_example_length": 2000,
    "date_range_days": 1095
  },
  "verbose": false,
  "debug": false
}
```

Environment variables override config file settings:
- `OLLAMA_HOST`: Ollama server URL
- `OLLAMA_MODEL`: Default model name
- `ZENDESK_VERBOSE`: Enable verbose logging (true/false)
- `ZENDESK_DEBUG`: Enable debug logging (true/false)

## File Formats

### Input Formats
The tools accept multiple input formats:

**Single JSON object:**
```json
{
  "id": 123,
  "comments": [
    {
      "id": 456,
      "body": "Cannot access my account",
      "plain_body": "Cannot access my account", 
      "created_at": "2023-01-01T10:00:00Z",
      "type": "Comment",
      "public": true,
      "author_id": 789
    }
  ]
}
```

**JSON array:**
```json
[
  {"id": 123, "comments": [...]},
  {"id": 124, "comments": [...]}
]
```

**JSONL (one JSON object per line):**
```
{"id": 123, "comments": [...]}
{"id": 124, "comments": [...]}
```

### Output Format

The classifier outputs CSV with these columns:
- `ticket_index`: Sequential ticket number
- `prima_data`: Earliest comment date
- `categoria_heuristica`: Heuristic classification result
- `categoria_finale`: Final classification (heuristic or Ollama)
- `categorie_trovate`: All matched categories
- `heuristic_triggers`: Regex patterns that matched
- `estratto`: Truncated ticket text
- `engine`: Classification engine used (heuristic/ollama)
- `source`: Classification source/reason
- `confidence`: Ollama confidence score (if available)

## Categories

Default Italian categories for classification:
- **Accesso/Login**: Login, access, passwords, 2FA issues
- **Registrazione**: Account registration and setup
- **Pagamenti/Fatturazione**: Payments, billing, invoices
- **Bug/Errore di sistema**: System errors, crashes, bugs
- **Usabilità/UX**: Interface, usability, form issues
- **Richiesta informazioni**: Information requests, guides
- **Altro**: Miscellaneous or unclassified tickets

## Architecture

```
zendesk_utils.py          # Shared utilities and base classes
├── TicketProcessor       # Base class for ticket operations
├── OllamaClient         # Async Ollama API client with pooling
├── TextPreprocessor     # Text cleaning and normalization
└── Validation utilities  # File and data validation

config.py                 # Configuration management
├── OllamaConfig         # Ollama client settings
├── ClassifierConfig     # Classification parameters
├── SynthesizerConfig    # Generation parameters
└── AppConfig           # Main application config

zendesk_ai_classifier.py  #  classifier
└── TicketClassifier    # Main classification logic

zendesk_synthesizer.py #  synthesizer
└── TicketSynthesizer   # Main generation logic
```

## Error Handling

The scripts include comprehensive error handling:

- **Connection errors**: Automatic retry with exponential backoff
- **JSON parsing errors**: Graceful fallback to alternative formats
- **Model availability**: Pre-flight checks before processing
- **Memory errors**: Streaming processing for large datasets
- **Validation errors**: Clear error messages with suggestions

## Development

To extend or modify the tools:

1. **Add new categories**: Update `config.py` classifier categories
2. **Custom preprocessing**: Extend `TextPreprocessor` in `zendesk_utils.py`  
3. **New output formats**: Modify save methods in classifier/synthesizer
4. **Additional models**: The code works with any Ollama-compatible model

## Troubleshooting

**Ollama connection issues:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if not running
ollama serve
```

**Memory issues with large datasets:**
- Reduce `batch_size` in configuration
- Use streaming processing (automatically enabled)
- Process files in chunks

**Performance optimization:**
- Increase `batch_size` for faster processing (if memory allows)
- Enable caching for repeated classifications
- Use SSD storage for better I/O performance# zendesk_ai_classifier
