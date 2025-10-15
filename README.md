# Hybris to Spring Boot Migration Tool

## How to Run

1. **Setup environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```

2. **Set API key:**
   ```bash
   set GEMINI_API_KEY=your_api_key_here
   ```

3. **Create knowledge base:**
   ```bash
   python build_know.py --src "Your Project Path" --out "knowledge_base"
   ```

4. **Run migration:**
   ```bash
   python generate_spring.py --kb "knowledge_base" --out "spring_output"
   ```

5. **Fix project structure structure (optional):**
   ```bash
   python fix_project_structure.py
   ```