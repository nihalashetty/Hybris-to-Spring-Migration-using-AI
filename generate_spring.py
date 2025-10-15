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
GEN_MAX_TOKENS = 3000
GEN_TEMPERATURE = 0.0
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

    def generate(self, prompt: str, max_tokens: int = GEN_MAX_TOKENS, temperature: float = GEN_TEMPERATURE, timeout: int = 120):
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
        return self._call_api(payload, timeout=timeout)

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
                if "content" in candidate and "parts" in candidate["content"]:
                    parts = candidate["content"]["parts"]
                    if parts and len(parts) > 0:
                        return parts[0].get("text", "")
        
        # fallback: return raw body
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

    # prepare snippets block
    snippet_block = ""
    for s in snippets:
        meta = s.get("meta", {})
        path = meta.get("path", "<unknown>")
        snippet = s.get("doc", "")[:6000]  # Increased context
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
- Use Spring Boot 3, Java 17+ features acceptable, @RestController for web endpoints, @Service for service layer.
- CRITICAL: Keep method names and signatures EXACTLY as in the original code.
- For controllers: Implement ALL HTTP endpoints from the original with proper REST responses (ResponseEntity<T>).
- For services: Implement ALL methods from the original with same signatures.
- If this is the first cluster (cluster_id=0), also generate: src/main/java/{{package|replace('.', '/')}}/MigratedApplication.java with @SpringBootApplication
- Do NOT invent database schemas. Where DB knowledge is missing, add a TODO comment with link to original file path.
- Output must be strictly in this format: 
  // FILE: src/main/java/{{package|replace('.', '/')}}/<ClassName>.java
  <Java source code>
  (repeat for each file)
- At the end of the output produce a JSON block header named // MIGRATION_MAPPING with an array of objects { "old": "<old path>", "new": "<relative new path>", "notes": "<free text risk notes>" }.

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
def generate(kb_path: Path, out_path: Path, gemini_key: str, skip_compile: bool = False):
    kb = load_kb(kb_path)
    client = PersistentClient(path=str(kb_path / CHROMA_DIRNAME))
    coll = client.get_collection("code_snippets")
    gemini = GeminiClient(gemini_key)
    model = SentenceTransformer(EMBED_MODEL)

    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "logs").mkdir(exist_ok=True)
    mapping = []

    clusters = kb.get("clusters", [])
    summaries = kb.get("summaries", {})

    logging.info("Starting generation for %d clusters", len(clusters))
    for ci, cluster in enumerate(tqdm(clusters, desc="clusters")):
        cluster_id = cluster.get("cluster_id", ci)
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
        logging.info("🔍 Retrieved %d code snippets for cluster %s", len(snippets), cluster_id)
        for i, snippet in enumerate(snippets):
            meta = snippet.get("meta", {})
            path = meta.get("path", "<unknown>")
            doc_length = len(snippet.get("doc", ""))
            logging.info("   📄 Snippet %d: %s (%d chars)", i+1, path, doc_length)
        
        prompt = make_cluster_prompt(cluster, cluster_id, summaries, snippets)

        # call Gemini to generate
        try:
            logging.info("📤 Sending prompt to Gemini for cluster %s (prompt length: %d chars)", cluster_id, len(prompt))
            logging.info("📋 Prompt preview: %s...", prompt[:200])
            gen_text = gemini.generate(prompt)
            logging.info("📥 Received response from Gemini for cluster %s (response length: %d chars)", cluster_id, len(gen_text))
            logging.info("📄 Response preview: %s...", gen_text[:300])
        except Exception as e:
            logging.error("Gemini generation failed for cluster %s: %s", cluster_id, e)
            continue

        files, mapping_json = parse_generated_files(gen_text)
        logging.info("🔍 Parsed %d files from Gemini response for cluster %s", len(files), cluster_id)
        for i, (file_path, content) in enumerate(files):
            logging.info("   📁 File %d: %s (%d lines)", i+1, file_path, len(content.splitlines()))
        
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

        # write files to a per-cluster package folder
        created_files = write_files_to_output(out_path, files, cluster_package_root=f"cluster_{cluster_id}")
        logging.info("✅ Successfully wrote %d files for cluster %s", len(created_files), cluster_id)
        # record mapping
        map_entry = {"cluster_id": cluster_id, "created_files": created_files, "raw_mapping": mapping_json, "notes": ""}
        mapping.append(map_entry)

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
    generate(kb, out, gemini_key, skip_compile=args.skip_compile)
