#!/usr/bin/env python3
"""
Phase 2: Generator Engine (Hybris knowledge base -> Spring Boot project)

Usage:
  python generate_spring.py --kb /path/to/knowledge_base --out /path/to/spring_output [--skip-compile]

Environment:
  GEMINI_API_KEY must be set.
  Optionally set GEMINI_API_BASE to override the default Gemini endpoint.

Notes:
  - This script loads the chroma DB inside the knowledge base.
  - It iterates clusters; for each cluster it retrieves top-K snippets and calls Gemini to generate Java files.
  - It then attempts to compile with Maven (unless --skip-compile is used) and runs automatic repair cycles on compile errors.
"""

import argparse
import json
import os
import sys
import logging
import subprocess
from pathlib import Path
from typing import List, Dict
from time import sleep
import time
import random
from jinja2 import Template
from tqdm import tqdm

# chroma + embeddings
import chromadb
from chromadb.config import Settings
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# HTTP
import requests

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------------- Config ----------------
EMBED_MODEL = "all-MiniLM-L6-v2"
CHROMA_DIRNAME = "chroma_db"
PACKAGE_ROOT = "com.company.migrated"
RETRIEVE_K = 12
GEN_MAX_TOKENS = 16000  # Further increased to handle complete responses
GEN_TEMPERATURE = 0.0
MAX_PROMPT_SIZE = 50000  # Maximum prompt size before compression
MAX_REPAIR_ITERS = 3
MVN_CMD = ["mvn", "-DskipTests", "package"]
# ----------------------------------------

# ---------------- Gemini client ----------------
class GeminiClient:
    """
    Minimal wrapper for Gemini-like REST API.
    If your provider is Vertex AI, adapt _call_api method to match their HTTP schema.
    """
    def __init__(self, api_key: str, api_base: str = None):
        self.api_key = api_key
        self.api_base = api_base or os.getenv("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment. Set GEMINI_API_KEY.")
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
        })

    def generate(self, prompt: str, max_tokens: int = GEN_MAX_TOKENS, temperature: float = GEN_TEMPERATURE, timeout: int = 120, max_retries: int = 3):
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature
            }
        }
        
        # Rate limiting: add small delay for small projects, larger for big ones
        delay = random.uniform(0.1, 0.5)  # Reduced from 0.5-2.0
        logging.info("Rate limiting: sleeping for %.2f seconds", delay)
        sleep(delay)
        
        # Retry logic with exponential backoff
        for attempt in range(max_retries):
            try:
                return self._call_api(payload, timeout=timeout)
            except Exception as e:
                if attempt < max_retries - 1:
                    backoff_delay = (2 ** attempt) + random.uniform(0, 1)
                    logging.warning("API call failed (attempt %d/%d), retrying in %.2f seconds: %s", 
                                  attempt + 1, max_retries, backoff_delay, str(e))
                    sleep(backoff_delay)
                else:
                    logging.error("API call failed after %d attempts: %s", max_retries, str(e))
                    raise

    def _call_api(self, payload: Dict, timeout: int = 120):
        """
        Call Google Gemini API with proper authentication.
        """
        # Add API key as query parameter for Google Gemini API
        url = f"{self.api_base}?key={self.api_key}"
        
        resp = self.session.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            logging.error("Gemini API error %s: %s", resp.status_code, resp.text)
            raise RuntimeError(f"Gemini API error {resp.status_code}")
        data = resp.json()
        
        # Parse Google Gemini API response format
        if isinstance(data, dict) and "candidates" in data:
            candidates = data["candidates"]
            if candidates and len(candidates) > 0:
                candidate = candidates[0]
                
                # Check for token limit issues
                if candidate.get("finishReason") == "MAX_TOKENS":
                    logging.error("Gemini hit MAX_TOKENS limit, response is truncated! This will cause incomplete generation.")
                
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and len(parts) > 0:
                        text = parts[0].get("text", "")
                        if text:
                            return text
                        else:
                            logging.warning("Empty response text from Gemini")
                            return ""
        
        # fallback: return raw body
        logging.warning("Could not parse Gemini response, returning raw text")
        return resp.text

# ---------------- Helpers ----------------
def load_kb(kb_path: Path):
    # load index, summaries, clusters, graph
    index = json.loads((kb_path / "index.json").read_text())
    ast_summaries = json.loads((kb_path / "ast_summaries.json").read_text())
    clusters = json.loads((kb_path / "clusters.json").read_text())
    graph = json.loads((kb_path / "feature_graph.json").read_text())
    return {"index": index, "summaries": ast_summaries, "clusters": clusters, "graph": graph}

def init_chroma(kb_path: Path):
    client = PersistentClient(path=str(kb_path / CHROMA_DIRNAME))
    coll = client.get_or_create_collection(name="code_snippets")
    model = SentenceTransformer(EMBED_MODEL)
    return client, coll, model

def retrieve_snippets(coll, query: str, n: int = RETRIEVE_K):
    # Chromadb query (text-based)
    try:
        res = coll.query(query_texts=[query], n_results=n)
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        ids = res["ids"][0]
        out = [{"id": ids[i], "doc": docs[i], "meta": metas[i]} for i in range(len(docs))]
        return out
    except Exception as e:
        logging.warning("Chroma retrieval failed: %s", e)
        return []

def filter_relevant_snippets(snippets: List[Dict], cluster_files: List[str], max_snippets: int = 8):
    """
    Filter snippets to only include those directly relevant to cluster files.
    This reduces prompt size and improves generation quality.
    """
    if len(snippets) <= max_snippets:
        return snippets
    
    relevant_snippets = []
    cluster_file_paths = set(cluster_files)
    
    # First pass: exact matches with cluster files
    for snippet in snippets:
        meta = snippet.get("meta", {})
        path = meta.get("path", "")
        if any(cluster_file in path for cluster_file in cluster_file_paths):
            relevant_snippets.append(snippet)
            if len(relevant_snippets) >= max_snippets:
                break
    
    # Second pass: if we need more, include by class/method relevance
    if len(relevant_snippets) < max_snippets:
        remaining = [s for s in snippets if s not in relevant_snippets]
        # Sort by document length (longer = more detailed)
        remaining.sort(key=lambda x: len(x.get("doc", "")), reverse=True)
        needed = max_snippets - len(relevant_snippets)
        relevant_snippets.extend(remaining[:needed])
    
    logging.info("Filtered %d snippets down to %d most relevant", len(snippets), len(relevant_snippets))
    return relevant_snippets

def compress_prompt(prompt: str, max_size: int) -> str:
    """
    Compress a prompt to fit within the maximum size limit.
    Prioritizes keeping instructions and file summaries, compresses snippets.
    """
    import re
    
    if len(prompt) <= max_size:
        return prompt
    
    # Split prompt into sections
    sections = re.split(r'(Instructions:|Relevant code snippets:|Cluster ID:)', prompt)
    
    # Keep instructions and cluster info (usually first parts)
    compressed_parts = []
    current_size = 0
    instructions_kept = False
    
    for i, section in enumerate(sections):
        section_size = len(section)
        
        # Always keep instructions and cluster info
        if "Instructions:" in section or "Cluster ID:" in section or not instructions_kept:
            if current_size + section_size <= max_size * 0.6:  # Reserve 60% for snippets
                compressed_parts.append(section)
                current_size += section_size
                if "Instructions:" in section:
                    instructions_kept = True
            continue
            
        # Compress snippet sections
        if "SNIPPET PATH:" in section:
            # Extract only essential parts: class definitions, method signatures
            lines = section.split('\n')
            compressed_lines = []
            in_snippet = False
            
            for line in lines:
                if 'SNIPPET PATH:' in line:
                    compressed_lines.append(line)
                    in_snippet = True
                elif in_snippet:
                    # Keep only essential lines
                    if (line.strip().startswith('public ') or 
                        line.strip().startswith('class ') or
                        line.strip().startswith('interface ') or
                        line.strip().startswith('@') or
                        '}' in line or '{' in line):
                        compressed_lines.append(line)
                    elif len(compressed_lines) < 30:  # Limit implementation details
                        compressed_lines.append(line)
            
            compressed_section = '\n'.join(compressed_lines[:50])  # Limit to 50 lines per snippet
            if current_size + len(compressed_section) <= max_size:
                compressed_parts.append(compressed_section)
                current_size += len(compressed_section)
        else:
            # For other sections, truncate if needed
            if current_size + section_size <= max_size:
                compressed_parts.append(section)
                current_size += section_size
            else:
                remaining = max_size - current_size
                if remaining > 100:  # Only add if meaningful space left
                    compressed_parts.append(section[:remaining])
                break
    
    result = ''.join(compressed_parts)
    logging.info("Compressed prompt from %d to %d chars (%.1f%% reduction)", 
                len(prompt), len(result), (1 - len(result)/len(prompt)) * 100)
    return result

def generate_cluster_hierarchically(cluster: Dict, cluster_id: int, summaries: Dict, snippets: List[Dict], gemini: GeminiClient):
    """
    Generate files for a cluster using hierarchical approach:
    1. Generate core services first
    2. Generate controllers using core services
    3. Generate models/DTOs
    """
    logging.info("Starting hierarchical generation for cluster %s", cluster_id)
    
    all_generated_files = []
    
    # Phase 1: Generate core services (ProductService, CartService, etc.)
    service_files = generate_core_services(cluster, cluster_id, summaries, snippets, gemini)
    all_generated_files.extend(service_files)
    
    # Phase 2: Generate models/DTOs
    model_files = generate_models(cluster, cluster_id, summaries, snippets, gemini, service_files)
    all_generated_files.extend(model_files)
    
    # Phase 3: Generate controllers using services and models
    controller_files = generate_controllers(cluster, cluster_id, summaries, snippets, gemini, service_files + model_files)
    all_generated_files.extend(controller_files)
    
    # Phase 4: Generate main application class if this is cluster 0
    if cluster_id == 0:
        main_app_file = generate_main_application(cluster_id, gemini)
        if main_app_file:
            all_generated_files.append(main_app_file)
    
    logging.info("Hierarchical generation complete for cluster %s: %d files generated", cluster_id, len(all_generated_files))
    return all_generated_files

def generate_core_services(cluster: Dict, cluster_id: int, summaries: Dict, snippets: List[Dict], gemini: GeminiClient):
    """Generate core service classes (ProductService, CartService, etc.)"""
    logging.info("🔧 Generating core services for cluster %s", cluster_id)
    
    # Filter snippets to service-related files
    service_snippets = [s for s in snippets if 'service' in s.get('meta', {}).get('path', '').lower()]
    
    prompt = f"""Generate ONLY the core service classes for cluster {cluster_id}.

Focus on:
- Service interfaces and implementations
- Business logic methods
- Data access patterns

Generate files in this format:
// FILE: src/main/java/com/company/migrated/cluster_{cluster_id}/service/ServiceName.java
[Service code]

Do not generate controllers or models yet."""
    
    # Add relevant snippets
    for s in service_snippets[:4]:  # Limit to 4 service snippets
        prompt += f"\n-- SNIPPET: {s.get('meta', {}).get('path', '')}\n{s.get('doc', '')[:3000]}\n"
    
    try:
        response = gemini.generate(prompt, max_tokens=4000)
        files, _ = parse_generated_files(response)
        logging.info("Generated %d service files", len(files))
        return files
    except Exception as e:
        logging.error("Failed to generate services: %s", e)
        return []

def generate_models(cluster: Dict, cluster_id: int, summaries: Dict, snippets: List[Dict], gemini: GeminiClient, existing_files: List):
    """Generate model/DTO classes"""
    logging.info("Generating models/DTOs for cluster %s", cluster_id)
    
    prompt = f"""Generate ONLY the model/DTO classes for cluster {cluster_id}.

Focus on:
- Data transfer objects (DTOs)
- Entity classes
- Value objects

Generate files in this format:
// FILE: src/main/java/com/company/migrated/cluster_{cluster_id}/model/ModelName.java
[Model code]

Existing service files context:
{str(existing_files)[:1000]}"""
    
    try:
        response = gemini.generate(prompt, max_tokens=3000)
        files, _ = parse_generated_files(response)
        logging.info("Generated %d model files", len(files))
        return files
    except Exception as e:
        logging.error("Failed to generate models: %s", e)
        return []

def generate_controllers(cluster: Dict, cluster_id: int, summaries: Dict, snippets: List[Dict], gemini: GeminiClient, existing_files: List):
    """Generate controller classes using existing services and models"""
    logging.info("🎮 Generating controllers for cluster %s", cluster_id)
    
    # Filter snippets to controller-related files
    controller_snippets = [s for s in snippets if 'controller' in s.get('meta', {}).get('path', '').lower()]
    
    prompt = f"""Generate ONLY the controller classes for cluster {cluster_id}.

Focus on:
- REST controllers with @RestController
- HTTP endpoint mappings
- Proper dependency injection
- Integration with existing services

Generate files in this format:
// FILE: src/main/java/com/company/migrated/cluster_{cluster_id}/controller/ControllerName.java
[Controller code]

Existing files context:
{str(existing_files)[:1500]}"""
    
    # Add controller snippets
    for s in controller_snippets[:4]:
        prompt += f"\n-- SNIPPET: {s.get('meta', {}).get('path', '')}\n{s.get('doc', '')[:3000]}\n"
    
    try:
        response = gemini.generate(prompt, max_tokens=4000)
        files, _ = parse_generated_files(response)
        logging.info("Generated %d controller files", len(files))
        return files
    except Exception as e:
        logging.error("Failed to generate controllers: %s", e)
        return []

def generate_main_application(cluster_id: int, gemini: GeminiClient):
    """Generate the main Spring Boot application class"""
    logging.info("Generating main application class for cluster %s", cluster_id)
    
    prompt = f"""Generate ONLY the main Spring Boot application class for cluster {cluster_id}.

Generate this file:
// FILE: src/main/java/com/company/migrated/cluster_{cluster_id}/MigratedApplication.java

The class should:
- Have @SpringBootApplication annotation
- Have a main method that starts SpringApplication
- Be in package com.company.migrated.cluster_{cluster_id}"""
    
    try:
        response = gemini.generate(prompt, max_tokens=1000)
        files, _ = parse_generated_files(response)
        if files:
            logging.info("Generated main application class")
            return files[0]
    except Exception as e:
        logging.error("Failed to generate main application: %s", e)
    return None

def make_cluster_prompt(cluster: Dict, cluster_id: int, summaries: Dict, snippets: List[Dict]):
    """
    Build a robust prompt for Gemini from cluster structural facts + retrieved snippets.

    We'll instruct Gemini to:
    - produce Spring Boot files using package com.company.migrated.<clustername>
    - keep method names where possible
    - annotate each file with header: // FILE: <path>
    - include mapping JSON at the end
    """
    cluster_name = f"cluster_{cluster_id}"
    package = f"{PACKAGE_ROOT}.{cluster_name}"
    # Build a short structural summary
    files = cluster.get("files", [])
    file_list_text = "\n".join([f"- {f}" for f in files[:200]])
    # gather brief summaries for each file
    file_summaries = []
    for f in files:
        s = summaries.get(f)
        if s:
            fs = s.get("summary")
            classes = [c.get("name") for c in fs.get("classes", [])] if fs else []
            file_summaries.append(f"{f} -> classes: {classes}")
    file_summaries_text = "\n".join(file_summaries[:200])

    # prepare snippets block with compression if needed
    snippet_block = ""
    for s in snippets:
        meta = s.get("meta", {})
        path = meta.get("path", "<unknown>")
        snippet = s.get("doc", "")
        
        # Compress snippet if it's too long
        if len(snippet) > 8000:
            # Keep method signatures and class structure, remove detailed implementation
            import re
            # Extract class definitions and method signatures
            class_matches = re.findall(r'public\s+class\s+\w+[^{]*\{[^}]*\}', snippet, re.DOTALL)
            method_matches = re.findall(r'public\s+[^{]*\{[^}]*\}', snippet, re.DOTALL)
            compressed = '\n'.join(class_matches[:3] + method_matches[:5])  # Keep top 3 classes and 5 methods
            snippet = compressed[:8000] if compressed else snippet[:8000]
        else:
            snippet = snippet[:6000]  # Original limit for smaller snippets
            
        snippet_block += f"\n-- SNIPPET PATH: {path}\n{snippet}\n"

    # load prompt template (inline)
    prompt_tpl = """You are an expert Java engineer. Convert the following Hybris backend code into idiomatic Spring Boot modules.

Cluster ID: {{cluster_id}}
Target package: {{package}}
Files in cluster:
{{file_list}}

File summaries:
{{file_summaries}}

Relevant code snippets (context):
{{snippet_block}}

Instructions:
- Generate Java source files for this cluster using package name exactly: {{package}}.
- PRIORITY ORDER: 1) Services with business logic, 2) Controllers with HTTP endpoints, 3) DTOs/Models
- CRITICAL: Keep method names and signatures EXACTLY as in the original code.
- For services: Implement ALL methods from the original with same signatures and business logic
- For controllers: Implement ALL HTTP endpoints from the original with proper REST responses (ResponseEntity<T>)
- Use Spring Boot 3, Java 17+ features acceptable, @RestController for web endpoints, @Service for service layer
- If this is the first cluster (cluster_id=0), also generate: src/main/java/{{package|replace('.', '/')}}/MigratedApplication.java with @SpringBootApplication
- Do NOT invent database schemas. Where DB knowledge is missing, add a TODO comment with link to original file path
- Output must be strictly in this format: 
  // FILE: src/main/java/{{package|replace('.', '/')}}/<ClassName>.java
  <Java source code>
  (repeat for each file)
- At the end of the output produce a JSON block header named // MIGRATION_MAPPING with an array of objects { "old": "<old path>", "new": "<relative new path>", "notes": "<free text risk notes>" }

Important:
- Be conservative: prefer generating clear scaffolding and TODO comments than guessing unknowns.
- Keep responses deterministic (temperature 0.0).
- Keep answers code-only (no extra commentary).

Now, generate the Java files for this cluster.
"""
    t = Template(prompt_tpl)
    return t.render(cluster_id=cluster_id, package=package, file_list=file_list_text, file_summaries=file_summaries_text, snippet_block=snippet_block)

def parse_generated_files(generated_text: str):
    """
    Expect generated_text to contain multiple blocks starting with:
      // FILE: src/main/java/...
    Return list of tuples (relative_path, content) and mapping_json if present.
    """
    files = []
    lines = generated_text.splitlines()
    cur_path = None
    cur_buf = []
    mapping_json = None
    for ln in lines:
        if ln.strip().startswith("// FILE:"):
            # flush previous
            if cur_path and cur_buf:
                files.append((cur_path.strip(), "\n".join(cur_buf).rstrip()+"\n"))
            cur_path = ln.split(":", 1)[1].strip()
            cur_buf = []
        elif ln.strip().startswith("// MIGRATION_MAPPING"):
            # next lines are JSON; gather until EOF
            # find the JSON substring
            rest = "\n".join(lines[lines.index(ln)+1:])
            try:
                mapping_json = json.loads(rest)
            except Exception:
                # try to extract a JSON block between braces
                jstart = rest.find("{")
                jend = rest.rfind("}")
                if jstart != -1 and jend != -1:
                    try:
                        mapping_json = json.loads(rest[jstart:jend+1])
                    except Exception:
                        mapping_json = None
            break
        else:
            if cur_path is not None:
                cur_buf.append(ln)
    # flush last
    if cur_path and cur_buf:
        files.append((cur_path.strip(), "\n".join(cur_buf).rstrip()+"\n"))
    return files, mapping_json

def write_files_to_output(out_dir: Path, files: List[tuple], cluster_package_root: str):
    created = []
    for rel_path, content in files:
        # sanitize and ensure file path is within out_dir
        target = out_dir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        # add MIGRATION header if not present
        header = f"// MIGRATION_FROM: {cluster_package_root}\n"
        if not content.startswith("// MIGRATION_FROM"):
            content = header + content
        target.write_text(content, encoding="utf-8")
        created.append(str(target.relative_to(out_dir)))
    return created

def run_maven_build(project_root: Path, timeout: int = 600):
    """
    Run mvn -DskipTests package and return (success, stdout+stderr)
    """
    if not (project_root / "pom.xml").exists():
        return False, "pom.xml not found"
    try:
        p = subprocess.run(MVN_CMD, cwd=str(project_root), capture_output=True, text=True, timeout=timeout)
        out = p.stdout + "\n" + p.stderr
        return (p.returncode == 0, out)
    except Exception as e:
        return False, str(e)

def extract_compile_errors(mvn_output: str):
    """
    Heuristically extract a short set of errors to include in repair prompt.
    """
    # grab lines containing error markers: [ERROR], cannot find symbol, method not found, package ... does not exist
    lines = mvn_output.splitlines()
    error_lines = [ln for ln in lines if "[ERROR]" in ln or "cannot find symbol" in ln or "package " in ln and "does not exist" in ln or "error:" in ln]
    # return last 4000 chars
    excerpt = "\n".join(error_lines[-200:])
    return excerpt if excerpt else mvn_output[:4000]

# ---------------- Main generation loop ----------------
def generate(kb_path: Path, out_path: Path, gemini_key: str, skip_compile: bool = False, hierarchical: bool = False):
    kb = load_kb(kb_path)
    client = PersistentClient(path=str(kb_path / CHROMA_DIRNAME))
    coll = client.get_collection("code_snippets")
    gemini = GeminiClient(gemini_key)
    model = SentenceTransformer(EMBED_MODEL)

    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "logs").mkdir(exist_ok=True)
    
    # Load existing progress if resuming
    progress_file = out_path / "generation_progress.json"
    completed_clusters = set()
    if progress_file.exists():
        try:
            progress_data = json.loads(progress_file.read_text())
            completed_clusters = set(progress_data.get("completed_clusters", []))
            logging.info("Resuming generation: %d clusters already completed", len(completed_clusters))
        except Exception as e:
            logging.warning("Failed to load progress file: %s", e)
    
    mapping = []

    clusters = kb.get("clusters", [])
    summaries = kb.get("summaries", {})

    logging.info("Starting generation for %d clusters", len(clusters))
    for ci, cluster in enumerate(tqdm(clusters, desc="clusters")):
        cluster_id = cluster.get("cluster_id", ci)
        
        # Skip if already completed
        if cluster_id in completed_clusters:
            logging.info("Skipping cluster %s (already completed)", cluster_id)
            continue
            
        files_in_cluster = cluster.get("files", [])
        logging.info("Generating cluster %s with %d files", cluster_id, len(files_in_cluster))

        # Build a natural language query for retrieval: combine file names, class names, and method signatures
        q_parts = []
        for f in files_in_cluster[:20]:
            s = summaries.get(f)
            if s:
                classes = s.get("summary", {}).get("classes", [])
                class_info = []
                for c in classes:
                    class_name = c.get("name", "")
                    methods = [m.get("name", "") for m in c.get("methods", [])]
                    if methods:
                        class_info.append(f"{class_name}(methods: {methods})")
                    else:
                        class_info.append(class_name)
                q_parts.append(f"{f} classes: {class_info}")
            else:
                q_parts.append(f)
        query_text = "\n".join(q_parts)[:3000]

        snippets = retrieve_snippets(coll, query_text, n=RETRIEVE_K)
        logging.info("Retrieved %d code snippets for cluster %s", len(snippets), cluster_id)
        
        # Apply smart filtering to reduce prompt size
        filtered_snippets = filter_relevant_snippets(snippets, files_in_cluster, max_snippets=8)
        
        for i, snippet in enumerate(filtered_snippets):
            meta = snippet.get("meta", {})
            path = meta.get("path", "<unknown>")
            doc_length = len(snippet.get("doc", ""))
            logging.info("   Snippet %d: %s (%d chars)", i+1, path, doc_length)
        
        prompt = make_cluster_prompt(cluster, cluster_id, summaries, filtered_snippets)
        
        # Check prompt size and compress if needed
        if len(prompt) > MAX_PROMPT_SIZE:
            logging.warning("Prompt size (%d chars) exceeds limit (%d), compressing...", len(prompt), MAX_PROMPT_SIZE)
            prompt = compress_prompt(prompt, MAX_PROMPT_SIZE)
            logging.info("Compressed prompt to %d chars", len(prompt))

        # Choose generation method
        if hierarchical:
            logging.info("Using hierarchical generation for cluster %s", cluster_id)
            try:
                files = generate_cluster_hierarchically(cluster, cluster_id, summaries, filtered_snippets, gemini)
                mapping_json = None  # Hierarchical doesn't generate mapping JSON yet
                logging.info("Hierarchical generation produced %d files for cluster %s", len(files), cluster_id)
            except Exception as e:
                logging.error("Hierarchical generation failed for cluster %s: %s", cluster_id, e)
                continue
        else:
            # Traditional one-shot generation
            try:
                logging.info("Sending prompt to Gemini for cluster %s (prompt length: %d chars)", cluster_id, len(prompt))
                logging.info("Prompt preview: %s...", prompt[:200])
                gen_text = gemini.generate(prompt)
                logging.info("📥 Received response from Gemini for cluster %s (response length: %d chars)", cluster_id, len(gen_text))
                logging.info("Response preview: %s...", gen_text[:300])
                
                files, mapping_json = parse_generated_files(gen_text)
                logging.info("Parsed %d files from Gemini response for cluster %s", len(files), cluster_id)
                
                if not files:
                    logging.warning("No files parsed from generator output for cluster %s. Saving raw output.", cluster_id)
                    (out_path / "logs" / f"cluster_{cluster_id}_raw.txt").write_text(gen_text)
                    # Try to extract any Java files even if parsing failed
                    import re
                    java_files = re.findall(r'// FILE: (.+?\.java)\n(.*?)(?=\n// FILE:|\n// MIGRATION_MAPPING|\Z)', gen_text, re.DOTALL)
                    if java_files:
                        for file_path, content in java_files:
                            target = out_path / file_path.strip()
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_text(content.strip(), encoding="utf-8")
                            logging.info("Extracted file: %s", file_path)
                    continue
                    
            except Exception as e:
                logging.error("Gemini generation failed for cluster %s: %s", cluster_id, e)
                continue

        # Log generated files
        for i, (file_path, content) in enumerate(files):
            logging.info("   File %d: %s (%d lines)", i+1, file_path, len(content.splitlines()))

        # write files to a per-cluster package folder
        created_files = write_files_to_output(out_path, files, cluster_package_root=f"cluster_{cluster_id}")
        logging.info("Successfully wrote %d files for cluster %s", len(created_files), cluster_id)
        
        # record mapping
        map_entry = {"cluster_id": cluster_id, "created_files": created_files, "raw_mapping": mapping_json, "notes": ""}
        mapping.append(map_entry)
        
        # Mark cluster as completed and save progress
        completed_clusters.add(cluster_id)
        progress_data = {
            "completed_clusters": list(completed_clusters),
            "last_updated": str(__import__("datetime").datetime.now(__import__("datetime").timezone.utc)),
            "total_clusters": len(clusters)
        }
        progress_file.write_text(json.dumps(progress_data, indent=2))
        logging.info("💾 Progress saved: %d/%d clusters completed", len(completed_clusters), len(clusters))

        # Optional compile + repair loop
        if not skip_compile:
            logging.info("Attempting to compile after generating cluster %s (this will compile entire project)", cluster_id)
            success, mvn_out = run_maven_build(out_path)
            (out_path / "logs" / f"cluster_{cluster_id}_mvn.log").write_text(mvn_out)
            if success:
                logging.info("Maven build succeeded after cluster %s generation", cluster_id)
            else:
                logging.warning("Maven build failed. Entering repair loop for cluster %s", cluster_id)
                # run repair iterations
                for iter_i in range(MAX_REPAIR_ITERS):
                    err_excerpt = extract_compile_errors(mvn_out)
                    # build repair prompt: include failing java file snippets (best effort) and the compiler errors
                    repair_prompt = f"""You are a Java expert. The generated Spring Boot project failed to compile. Below are the compiler error excerpts and the most relevant generated files. Fix only the portions necessary to make compilation work. Do not change broad structure. Provide only the updated file blocks in the same // FILE: <path> format.

Compiler errors:
{err_excerpt}

Most relevant file contents (first 6000 chars each):
"""
                    # attach up to 3 recently created files' contents
                    for cf in created_files[:3]:
                        p = out_path / cf
                        if p.exists():
                            repair_prompt += f"\n-- FILE: {cf}\n{p.read_text()[:6000]}\n"
                    repair_prompt += "\nNow produce the updated files."

                    try:
                        repair_text = gemini.generate(repair_prompt)
                    except Exception as e:
                        logging.error("Gemini repair call failed: %s", e)
                        break

                    new_files, _ = parse_generated_files(repair_text)
                    if new_files:
                        # overwrite those files
                        wrote = write_files_to_output(out_path, new_files, cluster_package_root=f"cluster_{cluster_id}")
                        logging.info("Wrote %d repaired files in iteration %d for cluster %s", len(wrote), iter_i+1, cluster_id)
                    else:
                        logging.warning("Repair iteration produced no files.")
                    # re-run mvn
                    success, mvn_out = run_maven_build(out_path)
                    (out_path / "logs" / f"cluster_{cluster_id}_mvn_iter_{iter_i}.log").write_text(mvn_out)
                    if success:
                        logging.info("Maven build succeeded after repair iteration %d for cluster %s", iter_i+1, cluster_id)
                        break
                if not success:
                    logging.error("Compilation still failing after %d repair attempts for cluster %s. See logs.", MAX_REPAIR_ITERS, cluster_id)
                    map_entry["notes"] = "compile_failed"
        # small sleep to avoid rate limiting
        sleep(0.5)

    # write mapping file
    (out_path / "migration_mapping.json").write_text(json.dumps(mapping, indent=2))
    # generation report
    report = {
        "generated_at": str(__import__("datetime").datetime.now(__import__("datetime").timezone.utc)),
        "clusters_generated": len(mapping),
        "out_path": str(out_path)
    }
    (out_path / "generation_report.md").write_text(json.dumps(report, indent=2))
    logging.info("Generation complete. Mapping written to %s", out_path / "migration_mapping.json")
    return out_path

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--kb", required=True, help="Path to knowledge_base (output of Phase 1)")
    p.add_argument("--out", required=True, help="Path to output Spring project folder")
    p.add_argument("--skip-compile", action="store_true", help="Skip maven compile & repair steps")
    p.add_argument("--hierarchical", action="store_true", help="Use hierarchical generation (recommended for large projects)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    kb = Path(args.kb)
    out = Path(args.out)
    gemini_key = "YOUR API KEY"
    if not gemini_key:
        logging.error("GEMINI_API_KEY environment variable is not set. Exiting.")
        sys.exit(1)
    if not kb.exists():
        logging.error("Knowledge base folder not found: %s", kb)
        sys.exit(1)
    out.mkdir(parents=True, exist_ok=True)
    generate(kb, out, gemini_key, skip_compile=args.skip_compile, hierarchical=args.hierarchical)
