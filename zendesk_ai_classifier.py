#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zendesk Ticket Classifier (Italiano) — Heuristics + Ollama
Version: v1.0

Usage:
  # Heuristics only
  python3 zendesk_classifier.py input.json --out results.csv

  # Ollama as fallback  
  python3 zendesk_classifier.py input.json --out results.csv --ollama --model mistral:7b

  # Force Ollama for all
  python3 zendesk_classifier.py input.json --out results.csv --ollama-for-all --model mistral:7b
"""

import argparse
import asyncio
import json
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import asdict

try:
    import pandas as pd
    import aiohttp
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required dependency: {e}", file=sys.stderr)
    print("Install with: pip install pandas aiohttp tqdm", file=sys.stderr)
    sys.exit(1)

from zendesk_utils import (
    TicketProcessor, OllamaClient, TextPreprocessor, 
    compile_regex_patterns, monitor_performance,
    EXIT_BAD_INPUT, EXIT_BAD_JSON, EXIT_OLLAMA, EXIT_CSV,
    logger
)
from config import AppConfig


class TicketClassifier(TicketProcessor):
    """ ticket classifier with async processing and caching"""
    
    def __init__(self, config: AppConfig):
        super().__init__(verbose=config.verbose)
        self.config = config
        self.preprocessor = TextPreprocessor()
        
        # Compile regex patterns once for performance
        self._compiled_patterns = {}
        for category, patterns in config.classifier.categories.items():
            if patterns:  # Skip empty pattern lists
                self._compiled_patterns[category] = compile_regex_patterns(tuple(patterns))
        
        # Initialize cache
        if config.classifier.enable_caching:
            self._classification_cache = {}
        
        self.user_prompt_template = """[TESTO TICKET]
{ticket}

[ISTRUZIONI]
- Analizza in italiano.
- Scegli SOLO una 'primary_category' dalla tassonomia.
- Rispondi SOLO con il JSON richiesto, niente spiegazioni."""
    
    @lru_cache(maxsize=128)
    def classify_heuristic_cached(self, text_hash: int) -> Tuple[str, List[str], List[str]]:
        """Cached heuristic classification for repeated text"""
        # This will be called with hash, but we need the actual text
        # We'll use this pattern in the non-cached version
        pass
    
    def classify_heuristic_detailed(self, text: str) -> Tuple[str, List[str], List[str]]:
        """
        Classify ticket using heuristic patterns with improved performance
        Returns: (primary_category, matched_categories, triggers)
        """
        # Check cache first
        if self.config.classifier.enable_caching:
            text_hash = hash(text)
            if text_hash in self._classification_cache:
                return self._classification_cache[text_hash]
        
        # Clean and normalize text
        cleaned_text = self.preprocessor.clean_text(text)
        
        matched_categories = []
        triggers = []
        found_categories = set()
        
        # Process patterns for each category
        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(cleaned_text)
                if match:
                    if category not in found_categories:
                        matched_categories.append(category)
                        found_categories.add(category)
                    triggers.append(f"{category}: '{match.group(0)}'")
                    break  # Only need first match per category
        
        # Primary category is first match, fallback to "Altro"
        primary = matched_categories[0] if matched_categories else "Altro"
        result = (primary, matched_categories, triggers)
        
        # Cache result
        if self.config.classifier.enable_caching:
            self._classification_cache[text_hash] = result
            
            # Limit cache size
            if len(self._classification_cache) > self.config.classifier.cache_size:
                # Remove oldest entries (simple FIFO)
                oldest_key = next(iter(self._classification_cache))
                del self._classification_cache[oldest_key]
        
        return result
    
    async def classify_with_ollama(self, text: str, ollama_client: OllamaClient) -> Optional[Dict[str, Any]]:
        """Classify single ticket using Ollama with validation"""
        system_prompt = self.config.classifier.build_system_prompt()
        user_prompt = self.user_prompt_template.format(ticket=text)
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        try:
            response = await ollama_client.generate_response(
                full_prompt,
                self.config.ollama.model
            )
            
            if not response:
                return None
            
            # Extract and validate JSON
            json_data = ollama_client.extract_json_response(response)
            if not json_data:
                return None
            
            # Validate primary category
            primary = json_data.get("primary_category", "").strip()
            if primary not in self.config.classifier.get_taxonomy():
                self.logger.warning(f"Invalid primary_category from Ollama: {primary}")
                return None
            
            # Validate secondary categories
            secondary = json_data.get("secondary_categories") or []
            if not isinstance(secondary, list):
                secondary = []
            
            # Validate confidence
            confidence = json_data.get("confidence")
            try:
                confidence = float(confidence) if confidence is not None else None
            except (TypeError, ValueError):
                confidence = None
            
            return {
                "primary_category": primary,
                "secondary_categories": secondary,
                "confidence": confidence,
                "raw": response[:500]  # Truncate raw response
            }
            
        except Exception as e:
            self.logger.error(f"Error in Ollama classification: {e}")
            return None
    
    async def process_batch_async(self, tickets: List[Dict[str, Any]], 
                                 use_ollama: bool = False, 
                                 ollama_for_all: bool = False) -> List[Dict[str, Any]]:
        """Process tickets in batches with async Ollama calls"""
        results = []
        
        if use_ollama or ollama_for_all:
            async with OllamaClient(
                host=self.config.ollama.host,
                timeout=self.config.ollama.timeout,
                max_retries=self.config.ollama.max_retries,
                backoff_factor=self.config.ollama.backoff_factor
            ) as ollama_client:
                
                # Check model availability
                if not await ollama_client.check_model(self.config.ollama.model):
                    raise Exception(f"Model {self.config.ollama.model} not available")
                
                # Process in batches
                batch_size = self.config.classifier.batch_size
                for i in range(0, len(tickets), batch_size):
                    batch = tickets[i:i + batch_size]
                    batch_results = await self._process_single_batch(
                        batch, ollama_client, use_ollama, ollama_for_all, i + 1
                    )
                    results.extend(batch_results)
        else:
            # Heuristic only processing
            for i, ticket in enumerate(tickets, 1):
                result = await self._process_single_ticket_heuristic(ticket, i)
                results.append(result)
        
        return results
    
    async def _process_single_batch(self, batch: List[Dict[str, Any]], 
                                   ollama_client: OllamaClient,
                                   use_ollama: bool, ollama_for_all: bool,
                                   start_idx: int) -> List[Dict[str, Any]]:
        """Process a single batch of tickets"""
        if ollama_for_all:
            # Create concurrent tasks for Ollama processing
            tasks = []
            for i, ticket in enumerate(batch):
                task = self._process_single_ticket_ollama_forced(
                    ticket, ollama_client, start_idx + i
                )
                tasks.append(task)
            
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        else:
            # Process with selective Ollama usage
            results = []
            ollama_tasks = []
            ollama_indices = []
            
            # First pass: heuristic classification
            for i, ticket in enumerate(batch):
                result = await self._process_single_ticket_heuristic(ticket, start_idx + i)
                results.append(result)
                
                # Queue for Ollama if heuristic result is "Altro"
                if use_ollama and result["categoria_finale"] == "Altro":
                    text, _ = self.extract_ticket_text(ticket)
                    task = self.classify_with_ollama(text, ollama_client)
                    ollama_tasks.append(task)
                    ollama_indices.append(i)
            
            # Second pass: process Ollama results
            if ollama_tasks:
                ollama_results = await asyncio.gather(*ollama_tasks, return_exceptions=True)
                
                for idx, ollama_result in zip(ollama_indices, ollama_results):
                    if isinstance(ollama_result, Exception):
                        self.logger.error(f"Ollama task failed: {ollama_result}")
                        continue
                    
                    if ollama_result and ollama_result.get("primary_category"):
                        # Update result with Ollama classification
                        self._update_result_with_ollama(results[idx], ollama_result, "ollama_fallback")
            
            return results
    
    async def _process_single_ticket_heuristic(self, ticket: Dict[str, Any], 
                                              ticket_index: int) -> Dict[str, Any]:
        """Process single ticket with heuristic classification only"""
        text, created_at = self.extract_ticket_text(ticket)
        
        # Heuristic classification
        heuristic_primary, heuristic_matched, heuristic_triggers = self.classify_heuristic_detailed(text)
        
        # Prepare result
        excerpt = self.preprocessor.truncate_with_ellipsis(
            text, self.config.classifier.text_truncate_length
        )
        
        return {
            "ticket_index": ticket_index,
            "prima_data": created_at,
            "categoria_heuristica": heuristic_primary,
            "categoria_finale": heuristic_primary,
            "categorie_trovate": ", ".join(heuristic_matched) if heuristic_matched else "",
            "heuristic_triggers": "; ".join(heuristic_triggers),
            "estratto": excerpt,
            "engine": "heuristic",
            "source": "heuristic",
            "ollama_model": "",
            "ollama_host": "",
            "ollama_reason": "",
            "confidence": ""
        }
    
    async def _process_single_ticket_ollama_forced(self, ticket: Dict[str, Any],
                                                  ollama_client: OllamaClient,
                                                  ticket_index: int) -> Dict[str, Any]:
        """Process single ticket with forced Ollama classification"""
        text, created_at = self.extract_ticket_text(ticket)
        
        # Get heuristic classification for comparison
        heuristic_primary, heuristic_matched, heuristic_triggers = self.classify_heuristic_detailed(text)
        
        # Try Ollama classification
        ollama_result = await self.classify_with_ollama(text, ollama_client)
        
        if ollama_result and ollama_result.get("primary_category"):
            final_primary = ollama_result["primary_category"]
            secondary = ollama_result.get("secondary_categories", [])
            
            # Merge secondary categories with heuristic matches
            all_matched = list(heuristic_matched)
            seen = set(all_matched)
            for sec in secondary:
                if sec not in seen and sec in self.config.classifier.get_taxonomy():
                    all_matched.append(sec)
                    seen.add(sec)
            
            result = self._create_base_result(
                ticket_index, created_at, text, heuristic_primary, 
                heuristic_matched, heuristic_triggers
            )
            
            self._update_result_with_ollama(result, ollama_result, "ollama_for_all")
            result["categoria_finale"] = final_primary
            result["categorie_trovate"] = ", ".join(all_matched) if all_matched else ""
            
            return result
        else:
            # Fallback to heuristic
            result = self._create_base_result(
                ticket_index, created_at, text, heuristic_primary,
                heuristic_matched, heuristic_triggers
            )
            result["source"] = "ollama_error_fallback_heuristic"
            result["ollama_reason"] = "ollama_error"
            return result
    
    def _create_base_result(self, ticket_index: int, created_at: str, text: str,
                           heuristic_primary: str, heuristic_matched: List[str],
                           heuristic_triggers: List[str]) -> Dict[str, Any]:
        """Create base result structure"""
        excerpt = self.preprocessor.truncate_with_ellipsis(
            text, self.config.classifier.text_truncate_length
        )
        
        return {
            "ticket_index": ticket_index,
            "prima_data": created_at,
            "categoria_heuristica": heuristic_primary,
            "categoria_finale": heuristic_primary,
            "categorie_trovate": ", ".join(heuristic_matched) if heuristic_matched else "",
            "heuristic_triggers": "; ".join(heuristic_triggers),
            "estratto": excerpt,
            "engine": "heuristic",
            "source": "heuristic",
            "ollama_model": "",
            "ollama_host": "",
            "ollama_reason": "",
            "confidence": ""
        }
    
    def _update_result_with_ollama(self, result: Dict[str, Any], 
                                  ollama_result: Dict[str, Any], source: str) -> None:
        """Update result with Ollama classification data"""
        result.update({
            "engine": "ollama",
            "source": source,
            "ollama_model": self.config.ollama.model,
            "ollama_host": self.config.ollama.host,
            "ollama_reason": source.replace("_", " "),
            "confidence": ollama_result.get("confidence", "")
        })
    
    @monitor_performance
    def save_results_to_csv(self, results: List[Dict[str, Any]], output_path: Path) -> None:
        """Save results to CSV with performance monitoring"""
        try:
            df = pd.DataFrame(results)
            
            # Calculate and display statistics
            total = len(df)
            distribution = df["categoria_finale"].value_counts().to_dict()
            
            self.logger.info(f"\n=== Classification Results ===")
            self.logger.info(f"Total tickets: {total}")
            self.logger.info("Category distribution:")
            for category, count in distribution.items():
                self.logger.info(f"  - {category}: {count}")
            
            # Count Ollama usage
            ollama_count = len(df[df["engine"] == "ollama"])
            if ollama_count > 0:
                self.logger.info(f"Tickets classified by Ollama: {ollama_count}")
            
            # Save to CSV
            df.to_csv(output_path, index=False, encoding="utf-8")
            self.logger.info(f"\nResults saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results to CSV: {e}")
            sys.exit(EXIT_CSV)


async def main():
    """Main async function"""
    parser = argparse.ArgumentParser(
        description=" Zendesk Ticket Classifier v5.0"
    )
    parser.add_argument("input", help="Path to JSON/JSONL input file")
    parser.add_argument("--out", default="classification_results.csv", 
                       help="Output CSV path")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--ollama", action="store_true",
                       help="Use Ollama as fallback for 'Altro' tickets")
    parser.add_argument("--ollama-for-all", action="store_true",
                       help="Force Ollama for ALL tickets")
    parser.add_argument("--model", help="Ollama model override")
    parser.add_argument("--host", help="Ollama host override") 
    parser.add_argument("--timeout", type=int, help="Ollama timeout override")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--batch-size", type=int, help="Batch size for processing")
    
    args = parser.parse_args()
    
    # Load configuration
    config = AppConfig.from_file(args.config) if args.config else AppConfig()
    
    # Apply command line overrides
    if args.model:
        config.ollama.model = args.model
    if args.host:
        config.ollama.host = args.host
    if args.timeout:
        config.ollama.timeout = args.timeout
    if args.verbose:
        config.verbose = True
    if args.batch_size:
        config.classifier.batch_size = args.batch_size
    
    # Initialize classifier
    classifier = TicketClassifier(config)
    
    try:
        # Load and parse tickets
        input_path = Path(args.input)
        raw_content = classifier.read_file(input_path)
        tickets = classifier.parse_tickets(raw_content)
        
        if not tickets:
            classifier.logger.error("No valid tickets found")
            sys.exit(EXIT_BAD_JSON)
        
        classifier.logger.info(f"Processing {len(tickets)} tickets...")
        
        # Process tickets with progress bar
        start_time = time.time()
        with tqdm(total=len(tickets), desc="Processing tickets") as pbar:
            # We'll update progress in batches
            results = await classifier.process_batch_async(
                tickets,
                use_ollama=args.ollama,
                ollama_for_all=args.ollama_for_all
            )
            pbar.update(len(tickets))
        
        processing_time = time.time() - start_time
        classifier.logger.info(f"Processing completed in {processing_time:.2f}s")
        
        # Save results
        output_path = Path(args.out)
        classifier.save_results_to_csv(results, output_path)
        
    except Exception as e:
        classifier.logger.error(f"Fatal error: {e}")
        sys.exit(EXIT_BAD_INPUT)


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
