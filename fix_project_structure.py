#!/usr/bin/env python3
"""
Script to reorganize the generated Spring Boot project structure
from cluster-based to proper layered architecture
"""

import shutil
from pathlib import Path
import re

def reorganize_spring_structure():
    """Reorganize the generated files into proper Spring Boot structure"""
    
    source_dir = Path("spring_output/src/main/java/com/company/migrated")
    backup_dir = Path("spring_output_backup")
    
    print("Reorganizing Spring Boot project structure...")
    
    # Create backup
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    shutil.copytree("spring_output", backup_dir)
    print(f"Created backup at {backup_dir}")
    
    # Create new structure
    new_structure = {
        "controller": [],
        "service": [],
        "dto": [],
        "model": [],
        "main": []
    }
    
    # Scan existing files and categorize them
    for cluster_dir in ["cluster_0", "cluster_1"]:
        cluster_path = source_dir / cluster_dir
        if not cluster_path.exists():
            continue
            
        for file_path in cluster_path.iterdir():
            if file_path.is_file() and file_path.suffix == '.java':
                filename = file_path.name
                
                # Categorize files
                if "Controller" in filename:
                    new_structure["controller"].append((file_path, filename))
                elif "Service" in filename:
                    new_structure["service"].append((file_path, filename))
                elif "DTO" in filename or "Request" in filename:
                    new_structure["dto"].append((file_path, filename))
                elif filename == "Product.java" or filename == "MigratedApplication.java":
                    new_structure["model"].append((file_path, filename))
                else:
                    new_structure["dto"].append((file_path, filename))  # Default to DTO
    
    # Handle main Application.java
    main_app = source_dir / "Application.java"
    if main_app.exists():
        new_structure["main"].append((main_app, "Application.java"))
    
    # Create new directory structure
    new_dirs = {
        "controller": source_dir / "controller",
        "service": source_dir / "service", 
        "dto": source_dir / "dto",
        "model": source_dir / "model"
    }
    
    for dir_path in new_dirs.values():
        dir_path.mkdir(exist_ok=True)
    
    # Move and update files
    moved_files = []
    
    for category, files in new_structure.items():
        if category == "main":
            continue  # Handle main separately
            
        target_dir = new_dirs[category]
        
        for source_file, filename in files:
            target_file = target_dir / filename
            
            # Read and update package declaration
            content = source_file.read_text(encoding="utf-8")
            
            # Update package declaration
            new_package = f"com.company.migrated.{category}"
            content = re.sub(
                r'package\s+com\.company\.migrated\.cluster_\d+;',
                f'package {new_package};',
                content
            )
            
            # Update imports if needed
            content = re.sub(
                r'import\s+com\.company\.migrated\.cluster_\d+\.',
                f'import com.company.migrated.',
                content
            )
            
            # Write to new location
            target_file.write_text(content, encoding="utf-8")
            moved_files.append((source_file, target_file))
            print(f"Moved {source_file.name} to {target_dir.name}/")
    
    # Handle main application class
    if new_structure["main"]:
        main_file, _ = new_structure["main"][0]
        content = main_file.read_text(encoding="utf-8")
        
        # Update package to root
        content = re.sub(
            r'package\s+com\.company\.migrated(?:\.cluster_\d+)?;',
            'package com.company.migrated;',
            content
        )
        
        # Update component scan if needed
        if '@SpringBootApplication' in content:
            content = re.sub(
                r'@SpringBootApplication',
                '@SpringBootApplication(scanBasePackages = "com.company.migrated")',
                content
            )
        
        main_file.write_text(content, encoding="utf-8")
        print(f"Updated main Application.java")
    
    # Clean up old cluster directories
    for cluster_dir in ["cluster_0", "cluster_1"]:
        cluster_path = source_dir / cluster_dir
        if cluster_path.exists():
            shutil.rmtree(cluster_path)
            print(f"Removed old cluster directory: {cluster_dir}")
    
    print(f"\nReorganization complete!")
    print(f"New structure:")
    print(f"   ├── controller/ ({len(new_structure['controller'])} files)")
    print(f"   ├── service/ ({len(new_structure['service'])} files)")
    print(f"   ├── dto/ ({len(new_structure['dto'])} files)")
    print(f"   ├── model/ ({len(new_structure['model'])} files)")
    print(f"   └── Application.java")
    
    return len(moved_files)

if __name__ == "__main__":
    moved_count = reorganize_spring_structure()
    print(f"\nSuccessfully reorganized {moved_count} files!")
