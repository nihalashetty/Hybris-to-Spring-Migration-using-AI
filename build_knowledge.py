#!/usr/bin/env python3
"""
Phase 1: Knowledge Builder for Hybris -> Spring migration.

Usage:
    python build_knowledge.py --src /path/to/hybris_repo --out /path/to/knowledge_base

Outputs (in out dir):
  - index.json
  - ast_summaries.json
  - feature_graph.json
  - clusters.json
  - chroma_db/ (Chroma persisted DB)
  - analysis_report.md

Notes:
  - Requires tree-sitter Java grammar installed by the script if missing.
  - Assumes a Linux/macOS environment; adjust paths for Windows.
"""

import argparse
import json
import os
import sys
import shutil
from pathlib import Path
import logging
from datetime import datetime, timezone
from typing import Dict, List, Tuple
import re
import multiprocessing

# AST parsing
try:
    from tree_sitter import Language, Parser
except Exception as e:
    print("tree_sitter not found. Please install 'tree_sitter' package first.")
    raise

# embeddings & vector DB
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# graph & clustering
import networkx as nx

# parallel helper
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# ---------- Configuration ----------
JAVA_LANGUAGE_SO = "build/my-languages.so"
JAVA_LANG_REPO = "https://github.com/tree-sitter/tree-sitter-java"
EMBED_MODEL = "all-MiniLM-L6-v2"   # lightweight and practical
CHROMA_DIRNAME = "chroma_db"
MAX_SNIPPET_LINES = 400
# -----------------------------------

def ensure_tree_sitter_java():
    """Ensure tree-sitter Java language is available."""
    try:
        # Try to use the pre-built tree-sitter-java package
        import tree_sitter_java as ts_java
        logging.info("Using pre-built tree-sitter Java language")
        # Return a flag to indicate we're using the pre-built package
        return ("prebuilt", ts_java.language())
    except ImportError:
        # Fallback to building from source (deprecated method)
        so_path = Path(JAVA_LANGUAGE_SO)
        if so_path.exists():
            logging.info("Using existing tree-sitter language binary at %s", so_path)
            return ("file", str(so_path))

        # Build process: clone minimal grammar and compile
        import subprocess, tempfile
        logging.info("Building tree-sitter Java grammar (this may take a minute)...")
        temp_dir = Path("build/_tree_sitter_temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        gram_dir = temp_dir / "tree-sitter-java"
        if not gram_dir.exists():
            subprocess.run(["git", "clone", "--depth", "1", JAVA_LANG_REPO, str(gram_dir)], check=True)
        # compile
        Language.build_library(
            # store the library in the `build` directory
            str(so_path),
            [
                str(gram_dir)
            ]
        )
        logging.info("Built tree-sitter language at %s", so_path)
        return ("file", str(so_path))

# ---------- Parsing utilities ----------
def init_parser():
    lang_type, java_lang = ensure_tree_sitter_java()
    if lang_type == "prebuilt":
        # For pre-built package, the java_lang is already a language object
        JAVALANG = Language(java_lang)
    else:
        # For file-based approach
        JAVALANG = Language(java_lang, "java")
    parser = Parser()
    parser.language = JAVALANG
    return parser

def parse_java_file(parser: Parser, path: Path):
    """Return tree-sitter tree and raw text for a java file."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        text = path.read_text(encoding="latin-1", errors="ignore")
    tree = parser.parse(bytes(text, "utf8"))
    return tree, text

def node_text(source_bytes: bytes, node):
    return source_bytes[node.start_byte:node.end_byte].decode("utf8", errors="ignore")

def extract_basic_info_from_tree(tree, source_text: str) -> Dict:
    """
    Walk the tree to extract classes, methods, imports, package, annotations, and docstrings.
    This returns a compact summary dict.
    """
    b = source_text.encode("utf8")
    root = tree.root_node
    summary = {
        "package": None,
        "imports": [],
        "classes": [],
        "file_annotations": []
    }
    # Walk children
    for child in root.children:
        if child.type == "package_declaration":
            summary["package"] = node_text(b, child).replace("package", "").strip().rstrip(";")
        elif child.type == "import_declaration":
            summary["imports"].append(node_text(b, child).strip())
        elif child.type in ("class_declaration", "interface_declaration", "enum_declaration"):
            cls = parse_class_node(b, child)
            summary["classes"].append(cls)
        elif child.type == "modifiers":
            # File-level annotations
            ann = extract_annotations(b, child)
            summary["file_annotations"].extend(ann)
    return summary

def parse_class_node(source_bytes: bytes, node) -> Dict:
    """
    Parse a class node to extract name, methods, fields, extends/implements and annotations.
    """
    cls = {
        "name": None,
        "methods": [],
        "fields": [],
        "extends": [],
        "implements": [],
        "annotations": []
    }
    # find identifier child and parse class body
    for c in node.children:
        if c.type == "identifier" and cls["name"] is None:
            cls["name"] = node_text(source_bytes, c)
        elif c.type == "modifiers":
            ann = extract_annotations(source_bytes, c)
            cls["annotations"].extend(ann)
        elif c.type == "class_body":
            # Parse class body for methods and fields
            for body_child in c.children:
                if body_child.type == "method_declaration":
                    m = parse_method_node(source_bytes, body_child)
                    cls["methods"].append(m)
                elif body_child.type == "field_declaration":
                    flds = parse_field_node(source_bytes, body_child)
                    cls["fields"].extend(flds)
        elif c.type == "superclass":
            cls["extends"].append(node_text(source_bytes, c))
        elif c.type == "super_interfaces":
            cls["implements"].append(node_text(source_bytes, c))
    return cls

def extract_annotations(source_bytes: bytes, node) -> List[str]:
    anns = []
    for c in node.children:
        if c.type in ("annotation", "marker_annotation"):
            anns.append(node_text(source_bytes, c))
    return anns

def parse_method_node(source_bytes: bytes, node) -> Dict:
    m = {"name": None, "params": [], "return_type": None, "annotations": [], "start_byte": node.start_byte, "end_byte": node.end_byte}
    
    # Parse method declaration structure based on tree-sitter-java grammar
    for c in node.children:
        if c.type == "modifiers":
            m["annotations"].extend(extract_annotations(source_bytes, c))
        elif c.type == "type_identifier":
            # Return type
            m["return_type"] = node_text(source_bytes, c)
        elif c.type == "identifier":
            # Method name
            m["name"] = node_text(source_bytes, c)
        elif c.type == "formal_parameters":
            # Parameters
            params = []
            for p in c.named_children:
                params.append(node_text(source_bytes, p))
            m["params"] = params
        elif c.type == "method_header":
            # Legacy fallback for different grammar versions
            for hh in c.children:
                if hh.type == "identifier":
                    m["name"] = node_text(source_bytes, hh)
                elif hh.type == "formal_parameters":
                    params = []
                    for p in hh.named_children:
                        params.append(node_text(source_bytes, p))
                    m["params"] = params
                elif hh.type == "result":
                    m["return_type"] = node_text(source_bytes, hh)
    return m

def parse_field_node(source_bytes: bytes, node) -> List[Dict]:
    fields = []
    # field_declaration has type + variable_declarator(s)
    typ = None
    for c in node.children:
        if c.type == "type":
            typ = node_text(source_bytes, c)
        elif c.type == "variable_declarator":
            # identifier child
            for vx in c.children:
                if vx.type == "identifier":
                    fields.append({"name": node_text(source_bytes, vx), "type": typ})
    return fields

# ---------- Graph building ----------
def build_call_edges_from_text(code_text: str) -> List[Tuple[str,str]]:
    """
    Heuristic: find patterns like SomeClass.methodName( or this.methodName( or service.someMethod(
    We'll try to extract caller->callee edges as "callerSymbol" -> "calleeSymbol".
    This is fuzzy but good for graph building.
    """
    edges = []
    # find method calls
    # pattern: Identifier.token(  or identifier.method(
    pattern = re.compile(r'([A-Za-z_][A-Za-z0-9_\.]*)\s*\(')
    candidates = pattern.findall(code_text)
    # filter out language keywords and common patterns
    stop = set(["if", "for", "while", "switch", "return", "new", "throw", "catch", "super", "this"])
    cleaned = [c for c in candidates if c and c.split('.')[0] not in stop and len(c) > 1]
    # naive edges: if code contains "a.b(" -> edge from current file to 'a' or 'a.b'
    # We'll return edges as pairs (file_symbol, called_symbol)
    # For now we just return unique called symbols
    for c in set(cleaned):
        edges.append((None, c))  # caller is None for now; we'll attach file-level caller later
    return edges

# ---------- Embeddings & Chroma ----------
def init_chroma_client(kb_out: Path):
    # Use the new ChromaDB API
    client = chromadb.PersistentClient(path=str(kb_out / CHROMA_DIRNAME))
    
    # Always ensure we have a collection with the correct embedding function
    from chromadb.utils import embedding_functions
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    
    try:
        # Try to get existing collection
        coll = client.get_collection("code_snippets")
        logging.info("Using existing ChromaDB collection")
    except Exception:
        # Create new collection with embedding function
        coll = client.create_collection("code_snippets", embedding_function=embedding_function)
        logging.info("Created new ChromaDB collection with embedding function")
    
    return client, coll

# ---------- Worker for single file processing ----------
def process_java_file(args):
    """Process a single java file path (for ThreadPool)."""
    parser, path_str = args
    path = Path(path_str)
    try:
        tree, text = parse_java_file(parser, path)
    except Exception as e:
        logging.exception("Failed to parse %s: %s", path, e)
        return None
    summary = extract_basic_info_from_tree(tree, text)
    # short snippet = first N lines
    snippet = "\n".join(text.splitlines()[:MAX_SNIPPET_LINES])
    # call edges
    edges = build_call_edges_from_text(text)
    return {
        "path": str(path),
        "summary": summary,
        "snippet": snippet,
        "edges": edges
    }

# ---------- Clustering (graph-based) ----------
def cluster_feature_graph(graph: nx.Graph, max_clusters: int = 80):
    """
    Use community detection / modularity maximize to detect clusters.
    If networkx has community module, use greedy modularity.
    """
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(graph))
        clusters = []
        for i, c in enumerate(comms):
            clusters.append(list(c))
        # limit
        if len(clusters) > max_clusters:
            clusters = clusters[:max_clusters]
        return clusters
    except Exception:
        logging.warning("Community detection not available; fallback to simple connected components.")
        comps = list(nx.connected_components(graph))
        clusters = [list(c) for c in comps]
        return clusters[:max_clusters]

# ---------- Main builder ----------
def build_knowledge(src: Path, out: Path, max_workers:int= max(2, multiprocessing.cpu_count()-1)):
    src = src.resolve()
    out = out.resolve()
    logging.info("Starting knowledge build from %s into %s", src, out)

    # create out dirs
    if out.exists():
        logging.info("Cleaning existing knowledge base files at %s", out)
        # Remove specific files, but preserve ChromaDB directory
        for file_pattern in ["*.json", "*.md"]:
            for file_path in out.glob(file_pattern):
                file_path.unlink()
        # Remove logs directory if it exists
        logs_dir = out / "logs"
        if logs_dir.exists():
            shutil.rmtree(logs_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / CHROMA_DIRNAME).mkdir(exist_ok=True)

    parser = init_parser()
    # gather java files
    java_files = [str(p) for p in src.rglob("*.java")]
    logging.info("Found %d Java files", len(java_files))
    index = {"root": str(src), "java_files_count": len(java_files), "java_files": java_files, "generated_at": datetime.now(timezone.utc).isoformat()}
    (out / "index.json").write_text(json.dumps(index, indent=2))

    # process files in parallel
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futs = [exe.submit(process_java_file, (parser, p)) for p in java_files]
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                results.append(r)
    logging.info("Parsed %d java files into summaries", len(results))

    # write AST summaries
    ast_summaries = {}
    for r in results:
        ast_summaries[r["path"]] = {
            "summary": r["summary"],
            "snippet": r["snippet"][:2000],  # store trimmed snippet
            "edges_count": len(r["edges"])
        }
    (out / "ast_summaries.json").write_text(json.dumps(ast_summaries, indent=2))

    # build a graph: nodes = file paths, edges = call relationships (best-effort)
    G = nx.DiGraph()
    for r in results:
        p = r["path"]
        G.add_node(p)
    # attach edges: for each file, add edges to called symbols; we'll map called symbols to files by simple symbol match
    # index class names -> file
    class_to_file = {}
    for r in results:
        p = r["path"]
        sumy = r["summary"]
        for cls in sumy.get("classes", []):
            name = cls.get("name")
            if name:
                class_to_file.setdefault(name, []).append(p)
    # now map edges heuristically
    for r in results:
        caller = r["path"]
        code = r["snippet"]
        callees = [c[1] for c in r["edges"]]
        for callee_sym in callees:
            # try exact class match
            base = callee_sym.split('.')[0]
            targets = class_to_file.get(base, [])
            for t in targets:
                if t != caller:
                    G.add_edge(caller, t)
    # export graph nodes/edges
    graph_data = {"nodes": list(G.nodes), "edges": [{"from": u, "to": v} for u, v in G.edges()]}
    (out / "feature_graph.json").write_text(json.dumps(graph_data, indent=2))

    # cluster graph
    logging.info("Clustering feature graph...")
    clusters = cluster_feature_graph(G)
    # map clusters to file lists
    clusters_mapped = []
    for i, c in enumerate(clusters):
        clusters_mapped.append({"cluster_id": i, "files": c, "size": len(c)})
    (out / "clusters.json").write_text(json.dumps(clusters_mapped, indent=2))

    # embeddings: index class-level + method-level snippets into Chroma
    logging.info("Creating embeddings and populating ChromaDB...")
    client, coll = init_chroma_client(out)
    model = SentenceTransformer(EMBED_MODEL)
    docs = []
    ids = []
    metas = []
    for r in results:
        p = r["path"]
        summary = r["summary"]
        # add class-level doc
        for cls in summary.get("classes", []):
            cls_name = cls.get("name")
            doc = f"FILE: {p}\nCLASS: {cls_name}\nANNOTATIONS: {cls.get('annotations')}\nMETHODS: {[m.get('name') for m in cls.get('methods', [])]}\nFIELDS: {cls.get('fields')}"
            idd = f"{p}::class::{cls_name}"
            docs.append(doc)
            ids.append(idd)
            metas.append({"path": p, "type": "class", "class": cls_name})
            # also method-level
            for m in cls.get("methods", []):
                mname = m.get("name") or "<anon>"
                snippet_text = ""
                # extract method text from original file (approx using byte offsets)
                # For speed, just add a placeholder describing signature
                snippet_text = f"FILE: {p}\nCLASS: {cls_name}\nMETHOD: {mname}\nPARAMS: {m.get('params')}\nANNOTATIONS: {m.get('annotations')}"
                mid = f"{p}::method::{cls_name}::{mname}"
                docs.append(snippet_text)
                ids.append(mid)
                metas.append({"path": p, "type": "method", "class": cls_name, "method": mname})
    logging.info("Prepared %d documents for ChromaDB insertion", len(docs))
    if docs:
        # Add documents to existing collection (avoid recreation)
        try:
            logging.info("Attempting to add documents to existing collection...")
            coll.add(ids=ids, documents=docs, metadatas=metas)
            logging.info("Successfully added %d documents to existing collection", len(docs))
            
            # Verify insertion worked
            count_after = coll.count()
            logging.info("Collection count after insertion: %d", count_after)
            if count_after == 0:
                logging.error("INSERTION FAILED: Collection count is still 0 after adding documents!")
                raise Exception("ChromaDB insertion failed - no documents persisted")
                
        except Exception as e:
            logging.warning("Failed to add to existing collection, recreating: %s", e)
            # Only recreate if absolutely necessary
            try:
                client.delete_collection("code_snippets")
                logging.info("Deleted existing collection")
            except:
                pass
            from chromadb.utils import embedding_functions
            embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=EMBED_MODEL
            )
            coll = client.create_collection("code_snippets", embedding_function=embedding_function)
            logging.info("Created new collection with embedding function")
            
            # Try insertion again
            coll.add(ids=ids, documents=docs, metadatas=metas)
            logging.info("Successfully added %d documents to new collection", len(docs))
            
            # Verify again
            count_after = coll.count()
            logging.info("Collection count after insertion: %d", count_after)
            if count_after == 0:
                logging.error("INSERTION STILL FAILED: Collection count is still 0 after recreating collection!")
    else:
        logging.warning("No documents prepared for ChromaDB insertion")
    
    # Final verification
    final_count = coll.count()
    logging.info("Final ChromaDB collection count: %d", final_count)
    if final_count == 0:
        logging.error("CRITICAL: ChromaDB collection is empty - embeddings were not stored!")

    # write an analysis report
    report = []
    report.append(f"Knowledge build report\nGenerated at: {datetime.now(timezone.utc).isoformat()}\n")
    report.append(f"Source root: {src}\nJava files parsed: {len(results)}\nClasses indexed: {len(ids)}")
    
    # Count methods and fields
    total_methods = sum(len(cls.get('methods', [])) for r in results for cls in r['summary'].get('classes', []))
    total_fields = sum(len(cls.get('fields', [])) for r in results for cls in r['summary'].get('classes', []))
    report.append(f"Total methods: {total_methods}\nTotal fields: {total_fields}")
    
    report.append("\nTop 20 largest clusters (by file count):")
    sorted_clusters = sorted(clusters_mapped, key=lambda x: x["size"], reverse=True)
    for c in sorted_clusters[:20]:
        report.append(f"- cluster {c['cluster_id']}: {c['size']} files")
    
    # Add detailed file breakdown
    report.append("\n\nFile Analysis:")
    for r in results:
        path = r['path']
        classes = r['summary'].get('classes', [])
        report.append(f"\n{path}:")
        for cls in classes:
            report.append(f"  Class: {cls.get('name', 'Unknown')}")
            report.append(f"    Methods: {len(cls.get('methods', []))}")
            report.append(f"    Fields: {len(cls.get('fields', []))}")
            report.append(f"    Annotations: {cls.get('annotations', [])}")
    
    try:
        (out / "analysis_report.md").write_text("\n".join(report))
        logging.info("Analysis report generated successfully")
    except Exception as e:
        logging.error("Failed to generate analysis report: %s", e)
        # Don't fail the entire process if report generation fails

    # final outputs already written
    logging.info("Knowledge base build complete. Output at: %s", out)
    return out

# ---------- CLI ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Hybris repo folder (root)")
    p.add_argument("--out", required=True, help="Output knowledge_base folder")
    p.add_argument("--max-workers", type=int, default=max(2, multiprocessing.cpu_count()-1))
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)
    if not src.exists():
        logging.error("Source path not found: %s", src)
        sys.exit(1)
    build_knowledge(src, out, max_workers=args.max_workers)
