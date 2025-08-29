#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for Zendesk ticket processing
Centralized configuration with validation and environment support
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class OllamaConfig:
    """Ollama client configuration"""
    host: str = "http://localhost:11434"
    #model: str = "gpt-oss:20b"
    model: str = "mistral:7b"
    timeout: int = 360
    max_retries: int = 3
    backoff_factor: float = 1.5
    
    def __post_init__(self):
        # Override with environment variables
        self.host = os.getenv('OLLAMA_HOST', self.host)
        self.model = os.getenv('OLLAMA_MODEL', self.model)
        
        # Validate timeout and retries
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")


@dataclass
class ClassifierConfig:
    """Configuration for ticket classifier"""
    categories: Dict[str, List[str]] = field(default_factory=lambda: {
        "Accesso/Login": [
            r"\blogin\b", r"\baccesso\b", r"\bentra(re)?\b", r"\bpassword\b",
            r"\b2FA\b", r"\bOTP\b", r"\bcodice\b", r"\bautenticazion(e|i)\b"
        ],
        "Registrazione": [
            r"\bregistrazion(e|i)\b", r"\bcrea(re)? account\b", r"\bnuov[oa] utenza\b"
        ],
        "Pagamenti/Fatturazione": [
            r"\bpagament[oi]\b", r"\bcarta\b", r"\bfattur[ae]\b", r"\bIBAN\b", 
            r"\baddebito\b", r"\bSEPA\b"
        ],
        "Bug/Errore di sistema": [
            r"\berrore\b", r"\bbug\b", r"\bcrash\b", r"\bblocc(a|o|ato)\b", 
            r"\bnon funziona\b", r"\beccezion(e|i)\b"
        ],
        "Usabilità/UX": [
            r"\bautocompletament(o|i)\b", r"\binterfaccia\b", r"\bmodulo\b", 
            r"\bform\b", r"\bpasso\b", r"\bstep\b"
        ],
        "Richiesta informazioni": [
            r"\binformazion(i|e)\b", r"\bchiariment[oi]\b", r"\bcome fare\b", 
            r"\bguida\b", r"\bdove trovo\b"
        ],
        "Altro": []
    })
    
    confidence_threshold: float = 0.7
    text_truncate_length: int = 220
    batch_size: int = 1  # For batch processing
    enable_caching: bool = True
    cache_size: int = 128
    
    def get_taxonomy(self) -> List[str]:
        """Get list of all category names"""
        return list(self.categories.keys())
    
    def build_system_prompt(self) -> str:
        """Build system prompt for Ollama classification"""
        bullets = "\n".join(f"- {cat}" for cat in self.get_taxonomy())
        return f"""Sei un classificatore di ticket di supporto in ITALIANO.
Devi assegnare UNA sola categoria primaria, scegliendola tra la seguente tassonomia fissa (usa ESATTAMENTE questi nomi):
{bullets}

RESTITUISCI SOLO un JSON valido in UNA riga, con questa forma esatta:
{{"primary_category":"<una delle categorie sopra>","secondary_categories":["..."],"confidence":0.0}}

Non usare sinonimi o varianti. Usa solo i nomi nella lista sopra."""


@dataclass
class SynthesizerConfig:
    """Configuration for synthetic ticket generator"""
    default_categories: List[str] = field(default_factory=lambda: [
        "Accesso/Login",
        "Registrazione", 
        "Pagamenti/Fatturazione",
        "Bug/Errore di sistema",
        "Usabilità/UX",
        "Richiesta informazioni",
        "Altro"
    ])
    
    default_num_tickets: int = 50
    max_example_length: int = 2000
    date_range_days: int = 3 * 365  # 3 years
    
    tone_hints: List[str] = field(default_factory=lambda: [
        "tono educato e dettagliato",
        "tono frettoloso e breve", 
        "tono aggressivo/frustrato",
        "tono confuso con dettagli contraddittori",
        "tono neutro e professionale"
    ])
    
    fallback_messages: List[str] = field(default_factory=lambda: [
        "Non riesco ad accedere, potete verificare? Grazie.",
        "La fattura ha importo errato.",
        "Il modulo di registrazione si blocca con errore 500.",
        "Non trovo dove cambiare numero di telefono.",
        "Perché il pagamento SEPA non va a buon fine?"
    ])
    
    def build_generation_prompt(self, example_text: str, categories: List[str]) -> str:
        """Build prompt for synthetic ticket generation"""
        bullets = "\n".join(f"- {c}" for c in categories)
        return f"""Sei un generatore di ticket di supporto (in ITALIANO) in formato tipo Zendesk.

CONTESTO DI STILE (esempio reale):
---
{example_text.strip()[:self.max_example_length]}
---

OBIETTIVO
- Genera UN singolo ticket (non una lista), simulando un utente che scrive al supporto.
- Il ticket deve essere coerente con il contesto "benefici/registrazione/account/pagamenti/bug/UX", ma varia toni e casistiche.
- Usa una delle categorie della seguente tassonomia (solo per GUIDARE IL CONTENUTO, non inserirla nel ticket!):
{bullets}

FORMATO DI USCITA (JSON, una riga):
{{
  "id": <numero intero univoco>,
  "comments": [
    {{
      "id": <numero intero>,
      "type": "Comment", 
      "author_id": <numero intero>,
      "body": "<testo utente>",
      "plain_body": "<testo utente>",
      "public": true,
      "created_at": "<ISO8601 con suffisso Z>"
    }}
    // eventuale secondo commento nello stesso ticket (opzionale)
  ],
  "count": <numero commenti>,
  "next_page": null,
  "previous_page": null
}}

REGOLE IMPORTANTI
- Scrivi SOLO il JSON, su UNA riga. Niente testo prima o dopo.
- Il testo deve essere naturale e verosimile; includi talvolta dettagli (link, importi, errori, browser).
- Varia i toni: educato, frettoloso, aggressivo, confuso.
- NON includere la categoria nel JSON. NON inventare campi extra.
- Le date devono essere realistiche (ultimi 3 anni) e coerenti tra i commenti.
- Evita dati sensibili reali: usa esempi plausibili."""


@dataclass
class AppConfig:
    """Main application configuration"""
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    synthesizer: SynthesizerConfig = field(default_factory=SynthesizerConfig)
    
    verbose: bool = False
    debug: bool = False
    
    def __post_init__(self):
        # Override with environment variables
        self.verbose = os.getenv('ZENDESK_VERBOSE', 'false').lower() == 'true'
        self.debug = os.getenv('ZENDESK_DEBUG', 'false').lower() == 'true'
    
    @classmethod
    def from_file(cls, config_path: str) -> 'AppConfig':
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Create config with defaults and update with file data
            config = cls()
            
            if 'ollama' in config_data:
                for key, value in config_data['ollama'].items():
                    if hasattr(config.ollama, key):
                        setattr(config.ollama, key, value)
            
            if 'classifier' in config_data:
                for key, value in config_data['classifier'].items():
                    if hasattr(config.classifier, key):
                        setattr(config.classifier, key, value)
            
            if 'synthesizer' in config_data:
                for key, value in config_data['synthesizer'].items():
                    if hasattr(config.synthesizer, key):
                        setattr(config.synthesizer, key, value)
            
            return config
            
        except FileNotFoundError:
            # Return default config if file doesn't exist
            return cls()
        except Exception as e:
            raise ValueError(f"Error loading config from {config_path}: {e}")
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        config_dict = {
            'ollama': {
                'host': self.ollama.host,
                'model': self.ollama.model,
                'timeout': self.ollama.timeout,
                'max_retries': self.ollama.max_retries,
                'backoff_factor': self.ollama.backoff_factor
            },
            'classifier': {
                'confidence_threshold': self.classifier.confidence_threshold,
                'text_truncate_length': self.classifier.text_truncate_length,
                'batch_size': self.classifier.batch_size,
                'enable_caching': self.classifier.enable_caching,
                'cache_size': self.classifier.cache_size
            },
            'synthesizer': {
                'default_num_tickets': self.synthesizer.default_num_tickets,
                'max_example_length': self.synthesizer.max_example_length,
                'date_range_days': self.synthesizer.date_range_days
            },
            'verbose': self.verbose,
            'debug': self.debug
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)


# Default configuration instance
DEFAULT_CONFIG = AppConfig()
