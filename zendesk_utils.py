#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for Zendesk ticket processing
Provides common functionality for classification and synthetic data generation
"""

import json
import logging
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import aiohttp
import asyncio
from functools import lru_cache

# Exit codes
EXIT_BAD_INPUT = 1
EXIT_BAD_JSON = 2
EXIT_OLLAMA = 3
EXIT_CSV = 4
EXIT_JSON = 3  # For JSON output errors in synthesizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OllamaError(Exception):
    """Custom exception for Ollama-related errors"""
    pass


class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass


def normalize_host(host: str) -> str:
    """Normalize Ollama host URL"""
    host = host.strip()
    if not host.startswith(("http://", "https://")):
        host = "http://" + host
    return host


def validate_file_path(path: Path, must_exist: bool = True) -> Path:
    """Validate file path with proper error handling"""
    if must_exist:
        if not path.exists():
            raise ValidationError(f"File not found: {path}")
        if not path.is_file():
            raise ValidationError(f"Not a regular file: {path}")
    return path


@lru_cache(maxsize=128)
def compile_regex_patterns(patterns: Tuple[str, ...]) -> List[re.Pattern]:
    """Compile regex patterns with caching for performance"""
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


class TicketProcessor:
    """Base class for ticket processing operations"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = logger
        if verbose:
            self.logger.setLevel(logging.DEBUG)
    
    def read_file(self, path: Path) -> str:
        """Read file with proper error handling and encoding"""
        try:
            path = validate_file_path(path)
            content = path.read_text(encoding='utf-8')
            if self.verbose:
                self.logger.info(f"Read {len(content)} characters from '{path}'")
            return content
        except Exception as e:
            self.logger.error(f"Error reading '{path}': {e}")
            sys.exit(EXIT_BAD_INPUT)
    
    def parse_tickets(self, raw_content: str) -> List[Dict[str, Any]]:
        """Parse tickets from JSON/JSONL format with improved error handling"""
        tickets = []
        
        # Try JSON first (single object or array)
        try:
            obj = json.loads(raw_content)
            if isinstance(obj, dict) and "comments" in obj:
                if self.verbose:
                    self.logger.info("Detected: single ticket JSON")
                return [obj]
            elif isinstance(obj, list):
                tickets = [t for t in obj if isinstance(t, dict) and "comments" in t]
                if self.verbose:
                    self.logger.info(f"Detected: JSON array with {len(tickets)} valid tickets")
                return tickets
        except json.JSONDecodeError:
            pass
        
        # Try JSONL format
        for line_num, line in enumerate(raw_content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "comments" in obj:
                    tickets.append(obj)
            except json.JSONDecodeError as e:
                self.logger.warning(f"Line {line_num}: Invalid JSON ({e}) - skipping")
        
        if self.verbose:
            self.logger.info(f"Detected: JSONL with {len(tickets)} valid tickets")
        
        if not tickets:
            raise ValidationError("No valid tickets found. Check format (dict/list with 'comments' or JSONL)")
        
        return tickets
    
    def extract_ticket_text(self, ticket: Dict[str, Any]) -> Tuple[str, str]:
        """Extract concatenated text and earliest creation date from ticket"""
        parts = []
        created_dates = []
        
        for comment in ticket.get("comments", []):
            body = comment.get("plain_body") or comment.get("body") or ""
            parts.append(body)
            
            created_at = comment.get("created_at")
            if created_at:
                created_dates.append(created_at)
        
        text = "\n\n".join(parts).strip()
        earliest_date = min(created_dates) if created_dates else ""
        
        return text, earliest_date


class OllamaClient:
    """Async Ollama client with connection pooling and retry logic"""
    
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 60, 
                 max_retries: int = 3, backoff_factor: float = 1.5):
        self.host = normalize_host(host)
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self._session: Optional[aiohttp.ClientSession] = None
        self.logger = logger
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            connector=aiohttp.TCPConnector(limit=10)  # Connection pooling
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
    async def check_model(self, model: str) -> bool:
        """Check if model is available on Ollama server"""
        url = f"{self.host.rstrip('/')}/api/tags"
        try:
            async with self._session.get(url, timeout=5) as response:
                response.raise_for_status()
                data = await response.json()
                models = [m.get("name") for m in data.get("models", []) if isinstance(m, dict)]
                
                if self.logger.level <= logging.DEBUG:
                    available = ', '.join(models) if models else '(none)'
                    self.logger.debug(f"Ollama OK on {self.host}. Available models: {available}")
                
                if model not in models:
                    self.logger.warning(f"Model '{model}' not found. Run: ollama pull {model}")
                    return False
                return True
        except Exception as e:
            raise OllamaError(f"Ollama unreachable on {self.host}: {e}")
    
    async def generate_response(self, prompt: str, model: str, **kwargs) -> Optional[str]:
        """Generate response from Ollama with retry logic"""
        url = f"{self.host.rstrip('/')}/api/generate"
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self._session.post(
                    url,
                    json={"model": model, "prompt": prompt, "stream": True, **kwargs}
                ) as response:
                    response.raise_for_status()
                    
                    buffer = ""
                    async for line in response.content:
                        line = line.decode('utf-8').strip()
                        if not line:
                            continue
                        
                        try:
                            chunk = json.loads(line)
                            content = chunk.get("response", "")
                            if content:
                                buffer += content
                            if chunk.get("done"):
                                break
                        except json.JSONDecodeError:
                            continue
                    
                    return buffer.strip()
                    
            except Exception as e:
                if attempt == self.max_retries:
                    self.logger.error(f"Ollama request failed after {self.max_retries + 1} attempts: {e}")
                    return None
                
                wait_time = self.backoff_factor ** attempt
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
        
        return None
    
    def extract_json_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Extract and parse JSON from Ollama response"""
        if not response:
            return None
        
        # Remove code fences
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", response, flags=re.IGNORECASE).strip()
        
        # Find JSON boundaries
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        
        if start == -1 or end == -1 or end <= start:
            return None
        
        candidate = cleaned[start:end + 1]
        
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            self.logger.debug(f"JSON parsing failed: {e}")
            return None


class TextPreprocessor:
    """Text preprocessing utilities for ticket classification"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text for processing"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Italian accents
        text = re.sub(r'[^\w\sàáâãäåæçèéêëìíîïñòóôõöøùúûüýÿ]', ' ', text)
        return text.strip().lower()
    
    @staticmethod
    def truncate_with_ellipsis(text: str, max_length: int = 220) -> str:
        """Truncate text with ellipsis if too long"""
        return text[:max_length] + "…" if len(text) > max_length else text


def generate_random_datetime(max_days: int = 3 * 365) -> datetime:
    """Generate random datetime within the last N days"""
    delta_days = __import__('random').randint(0, max_days)
    hours = __import__('random').randint(0, 23)
    minutes = __import__('random').randint(0, 59)
    base = datetime.utcnow() - timedelta(days=delta_days, hours=hours, minutes=minutes)
    return base


def format_iso_datetime(dt: datetime) -> str:
    """Format datetime as ISO8601 with Z suffix"""
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function execution time"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"{func.__name__} executed in {end_time - start_time:.2f}s")
        return result
    return wrapper