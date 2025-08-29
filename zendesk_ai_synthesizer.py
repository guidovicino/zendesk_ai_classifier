#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zendesk Synthetic Ticket Generator with Ollama
Version: v1.0

Usage:
  python3 zendesk_ai_synthesizer.py example.json --n 100 --out synthetic_tickets.json
"""

import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

try:
    import aiohttp
    from tqdm import tqdm
except ImportError as e:
    print(f"Missing required dependency: {e}", file=sys.stderr)
    print("Install with: pip install aiohttp tqdm", file=sys.stderr)
    sys.exit(1)

from zendesk_utils import (
    TicketProcessor, OllamaClient, generate_random_datetime, 
    format_iso_datetime, monitor_performance,
    EXIT_BAD_INPUT, EXIT_OLLAMA, EXIT_JSON,
    logger
)
from config import AppConfig


class TicketSynthesizer(TicketProcessor):
    """Synthetic ticket generator with async processing"""
    
    def __init__(self, config: AppConfig):
        super().__init__(verbose=config.verbose)
        self.config = config
        
        # Generate unique base ID for this session
        self.base_id = int(time.time())
        
        # Track generation statistics
        self.stats = {
            "generated": 0,
            "ollama_successful": 0,
            "fallback_used": 0,
            "errors": 0
        }
    
    def validate_example_content(self, content: str) -> str:
        """Validate and prepare example content for prompt"""
        if not content.strip():
            raise ValueError("Example file is empty")
        
        # Try to parse as JSON to validate format
        try:
            # Check if it's valid JSON (single object or array)
            json.loads(content)
        except json.JSONDecodeError:
            # Try JSONL format
            lines = [line.strip() for line in content.splitlines() if line.strip()]
            valid_lines = 0
            for line in lines[:5]:  # Check first 5 lines
                try:
                    json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError:
                    continue
            
            if valid_lines == 0:
                raise ValueError("Example file is not valid JSON or JSONL format")
        
        return content
    
    async def generate_batch_async(self, num_tickets: int, 
                                  example_content: str,
                                  categories: List[str]) -> List[Dict[str, Any]]:
        """Generate tickets in batches with async processing"""
        tickets = []
        batch_size = min(10, num_tickets)  # Limit concurrent requests
        
        async with OllamaClient(
            host=self.config.ollama.host,
            timeout=self.config.ollama.timeout,
            max_retries=self.config.ollama.max_retries,
            backoff_factor=self.config.ollama.backoff_factor
        ) as ollama_client:
            
            # Check model availability
            if not await ollama_client.check_model(self.config.ollama.model):
                raise Exception(f"Model {self.config.ollama.model} not available")
            
            # Process in batches with progress tracking
            with tqdm(total=num_tickets, desc="Generating tickets") as pbar:
                for i in range(0, num_tickets, batch_size):
                    batch_end = min(i + batch_size, num_tickets)
                    current_batch_size = batch_end - i
                    
                    # Create tasks for current batch
                    tasks = []
                    for j in range(current_batch_size):
                        ticket_id = self.base_id + i + j
                        task = self._generate_single_ticket(
                            ollama_client, example_content, categories, ticket_id
                        )
                        tasks.append(task)
                    
                    # Execute batch concurrently
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results and handle exceptions
                    for result in batch_results:
                        if isinstance(result, Exception):
                            self.logger.error(f"Ticket generation failed: {result}")
                            self.stats["errors"] += 1
                            # Create fallback ticket
                            ticket = self._create_fallback_ticket()
                        else:
                            ticket = result
                        
                        tickets.append(ticket)
                        pbar.update(1)
        
        return tickets
    
    async def _generate_single_ticket(self, ollama_client: OllamaClient,
                                     example_content: str, categories: List[str],
                                     ticket_id: int) -> Dict[str, Any]:
        """Generate a single ticket with enhanced prompt engineering"""
        
        # Build dynamic prompt with context variation
        base_prompt = self.config.synthesizer.build_generation_prompt(example_content, categories)
        
        # Add randomized context hints for variety
        tone_hint = random.choice(self.config.synthesizer.tone_hints)
        date_constraint = f"Date range: last {self.config.synthesizer.date_range_days // 365} years"
        additional_context = f"\n\nGeneration hints: {tone_hint}. {date_constraint}."
        
        full_prompt = base_prompt + additional_context
        
        # Generate response
        response = await ollama_client.generate_response(full_prompt, self.config.ollama.model)
        
        if response:
            # Try to extract and validate JSON
            ticket_data = ollama_client.extract_json_response(response)
            if ticket_data and self._validate_ticket_structure(ticket_data):
                # Post-process and normalize the ticket
                normalized_ticket = self._normalize_ticket_data(ticket_data, ticket_id)
                self.stats["generated"] += 1
                self.stats["ollama_successful"] += 1
                return normalized_ticket
        
        # Fallback if Ollama generation failed
        self.logger.warning(f"Ollama generation failed for ticket {ticket_id}, using fallback")
        self.stats["fallback_used"] += 1
        return self._create_fallback_ticket(ticket_id)
    
    def _validate_ticket_structure(self, ticket_data: Dict[str, Any]) -> bool:
        """Validate generated ticket structure"""
        try:
            # Check required fields
            if not isinstance(ticket_data, dict):
                return False
            
            if "comments" not in ticket_data:
                return False
            
            comments = ticket_data["comments"]
            if not isinstance(comments, list) or len(comments) == 0:
                return False
            
            # Validate each comment
            for comment in comments:
                if not isinstance(comment, dict):
                    return False
                
                required_fields = ["body", "plain_body", "created_at", "type", "public"]
                for field in required_fields:
                    if field not in comment:
                        return False
                
                # Validate date format
                try:
                    created_at = comment["created_at"]
                    if created_at.endswith("Z"):
                        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    else:
                        datetime.fromisoformat(created_at)
                except (ValueError, AttributeError):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Ticket validation failed: {e}")
            return False
    
    def _normalize_ticket_data(self, ticket_data: Dict[str, Any], ticket_id: int) -> Dict[str, Any]:
        """Normalize and enhance generated ticket data"""
        normalized = dict(ticket_data)
        
        # Ensure correct ticket ID
        normalized["id"] = ticket_id
        
        # Process comments
        comments = []
        for i, comment in enumerate(normalized.get("comments", [])):
            normalized_comment = dict(comment)
            
            # Ensure required fields have defaults
            normalized_comment["id"] = comment.get("id", random.randint(7000000000000, 7999999999999))
            normalized_comment["type"] = comment.get("type", "Comment")
            normalized_comment["public"] = bool(comment.get("public", True))
            normalized_comment["author_id"] = comment.get("author_id", random.randint(100000000, 999999999))
            
            # Normalize text content
            body_text = comment.get("plain_body") or comment.get("body") or ""
            normalized_comment["body"] = body_text
            normalized_comment["plain_body"] = body_text
            
            # Validate and normalize date
            created_at = comment.get("created_at", "")
            try:
                if created_at.endswith("Z"):
                    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                else:
                    dt = datetime.fromisoformat(created_at)
                normalized_comment["created_at"] = format_iso_datetime(dt)
            except (ValueError, AttributeError):
                # Generate random realistic date
                random_dt = generate_random_datetime(self.config.synthesizer.date_range_days)
                normalized_comment["created_at"] = format_iso_datetime(random_dt)
            
            comments.append(normalized_comment)
        
        # Sort comments by creation date
        comments.sort(key=lambda c: c.get("created_at", ""))
        
        # Update normalized ticket
        normalized["comments"] = comments
        normalized["count"] = len(comments)
        normalized["next_page"] = None
        normalized["previous_page"] = None
        
        return normalized
    
    def _create_fallback_ticket(self, ticket_id: Optional[int] = None) -> Dict[str, Any]:
        """Create fallback ticket when Ollama generation fails"""
        if ticket_id is None:
            ticket_id = self.base_id + self.stats["generated"]
        
        # Select random fallback message
        message = random.choice(self.config.synthesizer.fallback_messages)
        
        # Generate random recent date
        random_dt = generate_random_datetime(self.config.synthesizer.date_range_days)
        created_at = format_iso_datetime(random_dt)
        
        return {
            "id": ticket_id,
            "comments": [{
                "id": random.randint(7000000000000, 7999999999999),
                "type": "Comment",
                "author_id": random.randint(100000000, 999999999),
                "body": message,
                "plain_body": message,
                "public": True,
                "created_at": created_at
            }],
            "count": 1,
            "next_page": None,
            "previous_page": None
        }
    
    @monitor_performance
    def save_tickets_to_file(self, tickets: List[Dict[str, Any]], output_path: Path) -> None:
        """Save generated tickets to JSON file with performance monitoring"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tickets, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"\nGeneration completed successfully!")
            self.logger.info(f"Generated tickets: {len(tickets)}")
            self.logger.info(f"Ollama successful: {self.stats['ollama_successful']}")
            self.logger.info(f"Fallback used: {self.stats['fallback_used']}")
            self.logger.info(f"Errors encountered: {self.stats['errors']}")
            self.logger.info(f"Output saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving tickets to file: {e}")
            sys.exit(EXIT_JSON)
    
    def print_sample_tickets(self, tickets: List[Dict[str, Any]], num_samples: int = 3) -> None:
        """Print sample tickets for verification"""
        if not tickets or not self.verbose:
            return
        
        self.logger.info(f"\n=== Sample Generated Tickets ===")
        for i, ticket in enumerate(tickets[:num_samples]):
            self.logger.info(f"\nTicket {i+1} (ID: {ticket.get('id')}):")
            for comment in ticket.get("comments", []):
                text = comment.get("plain_body", "")[:100]
                date = comment.get("created_at", "")
                self.logger.info(f"  - {text}... (Date: {date})")


async def main():
    """Main async function"""
    parser = argparse.ArgumentParser(
        description=" Zendesk Synthetic Ticket Generator v2.0"
    )
    parser.add_argument("example", help="Example JSON/JSONL file for style context")
    parser.add_argument("--n", type=int, help="Number of tickets to generate")
    parser.add_argument("--out", default="synthetic_tickets.json", help="Output JSON file")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--categories", help="Comma-separated category list")
    parser.add_argument("--model", help="Ollama model override")
    parser.add_argument("--host", help="Ollama host override")
    parser.add_argument("--timeout", type=int, help="Ollama timeout override")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--sample", action="store_true", help="Show sample generated tickets")
    
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
    
    # Determine number of tickets to generate
    num_tickets = args.n if args.n else config.synthesizer.default_num_tickets
    
    # Parse categories
    if args.categories:
        categories = [cat.strip() for cat in args.categories.split(",") if cat.strip()]
    else:
        categories = config.synthesizer.default_categories
    
    # Initialize synthesizer
    synthesizer = TicketSynthesizer(config)
    
    try:
        # Load and validate example content
        example_path = Path(args.example)
        example_content = synthesizer.read_file(example_path)
        example_content = synthesizer.validate_example_content(example_content)
        
        synthesizer.logger.info(f"Generating {num_tickets} tickets using {len(categories)} categories...")
        synthesizer.logger.info(f"Categories: {', '.join(categories)}")
        
        # Generate tickets
        start_time = time.time()
        tickets = await synthesizer.generate_batch_async(num_tickets, example_content, categories)
        generation_time = time.time() - start_time
        
        synthesizer.logger.info(f"Generation completed in {generation_time:.2f}s")
        
        # Show samples if requested
        if args.sample:
            synthesizer.print_sample_tickets(tickets)
        
        # Save results
        output_path = Path(args.out)
        synthesizer.save_tickets_to_file(tickets, output_path)
        
    except Exception as e:
        synthesizer.logger.error(f"Fatal error: {e}")
        sys.exit(EXIT_BAD_INPUT)


if __name__ == "__main__":
    # Run async main
    asyncio.run(main())
