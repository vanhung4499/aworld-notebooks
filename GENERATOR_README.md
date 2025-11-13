# AWorld MAS Notebook Generator

Generate transparent, auditable Jupyter notebooks that demonstrate step-by-step agent execution for tasks from the GAIA benchmark dataset.

## Features

- **Full Transparency**: Every agent action, tool call, and state change is visible in separate cells
- **Live Execution**: Notebooks execute tasks in real-time with your MAS
- **MCP Visualization**: Shows MCP server tool calls and interactions
- **Answer Validation**: Automatic extraction and comparison with ground truth
- **Robust Path Detection**: Works from multiple directory contexts
- **Batch Generation**: Generate hundreds of notebooks efficiently

## Installation

### 1. Prerequisites

```bash
# Python 3.8+ required
python --version

# Install AWorld MAS framework
cd /path/to/AWorld
pip install -e .
```

### 2. Install Generator Dependencies

```bash
pip install nbformat pandas python-dotenv
```

### 3. Download GAIA Dataset

Download the GAIA benchmark dataset from [HuggingFace](https://huggingface.co/datasets/gaia-benchmark/GAIA):

```bash
# Option 1: Using git-lfs
git lfs install
git clone https://huggingface.co/datasets/gaia-benchmark/GAIA gaia_dataset

# Option 2: Manual download
# Download validation and test splits from HuggingFace web interface
mkdir -p gaia_dataset/validation
mkdir -p gaia_dataset/test
# Place metadata.jsonl and files in respective directories
```

### 4. Configure Environment

```bash
# Copy template
cp .env.template .env

# Edit .env and add your API keys
nano .env
```

**Required configuration:**
```bash
LLM_API_KEY=your_api_key_here  # OpenAI, Anthropic, etc.
GAIA_DATASET_PATH=./gaia_dataset
```

**Optional (for MCP tools):**
```bash
GOOGLE_API_KEY=your_google_key
GOOGLE_SEARCH_ENGINE_ID=your_search_id
E2B_API_KEY=your_e2b_key
```

## Usage

### Generate Single Notebook

**By Task ID:**
```bash
python generate_aworld_notebook.py --id 1-0-0-0-0
```

**By Index:**
```bash
python generate_aworld_notebook.py --index 0
```

**Specify split and output:**
```bash
python generate_aworld_notebook.py --id 1-0-0-0-0 \
    --split validation \
    --output-dir notebooks \
    --dataset-path ./gaia_dataset
```

### Batch Generation

**Generate range of tasks:**
```bash
# Generate notebooks for indices 0-9
python generate_aworld_notebook.py --start 0 --end 10

# Generate first 100 validation tasks
python generate_aworld_notebook.py --start 0 --end 100 --split validation

# Generate test split tasks
python generate_aworld_notebook.py --start 0 --end 50 --split test
```

**Generate all validation tasks:**
```bash
# GAIA validation has ~300 tasks
python generate_aworld_notebook.py --start 0 --end 300 --split validation
```

## Notebook Structure

Each generated notebook contains approximately 18-22 cells:

### 1. Task Display (2 cells)
- **Markdown**: Header and description
- **Code**: Task details, question, ground truth, metadata

### 2. Setup & Configuration (2 cells)
- **Markdown**: Setup section header
- **Code**: Path detection, imports, config loading, MCP initialization

### 3. Agent Initialization (2 cells)
- **Markdown**: Agent creation header
- **Code**: Create GAIA super agent with MCP servers

### 4. Task Execution (2 cells)
- **Markdown**: Execution header
- **Code**: Run task, capture results and trajectory

### 5. Trajectory Display (2 cells)
- **Markdown**: Trajectory analysis header
- **Code**: Step-by-step execution breakdown

### 6. Tool Calls (2 cells)
- **Markdown**: MCP tool calls header
- **Code**: Detailed tool execution logs

### 7. Agent Messages (2 cells)
- **Markdown**: LLM interactions header
- **Code**: Agent messages and LLM call details

### 8. Validation (2 cells)
- **Markdown**: Validation header
- **Code**: Answer extraction and comparison (✅/❌)

## Running Generated Notebooks

### Option 1: Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ directory and open a notebook
# Run cells sequentially or use "Run All"
```

### Option 2: JupyterLab

```bash
# Start JupyterLab
jupyter lab

# Open notebook and execute cells
```

### Option 3: Command Line

```bash
# Execute notebook and save output
jupyter nbconvert --to notebook --execute notebooks/task_1-0-0-0-0.ipynb \
    --output task_1-0-0-0-0_executed.ipynb
```

### Option 4: VS Code

```bash
# Open in VS Code with Jupyter extension
code notebooks/task_1-0-0-0-0.ipynb
```

## Troubleshooting

### Module Not Found Errors

**Problem:** `ModuleNotFoundError: No module named 'aworld'`

**Solutions:**
1. Install AWorld in editable mode:
   ```bash
   cd /path/to/AWorld
   pip install -e .
   ```

2. Verify installation:
   ```bash
   python -c "import aworld; print(aworld.__file__)"
   ```

3. Run notebook from AWorld root directory:
   ```bash
   cd /path/to/AWorld
   jupyter notebook notebooks/
   ```

### Config File Not Found

**Problem:** `.env` file or `mcp.json` not found

**Solutions:**
1. The notebook searches multiple paths automatically:
   - Current directory
   - Parent directory
   - AWorld root
   - `examples/gaia/` directory

2. Copy config files to notebook directory:
   ```bash
   cp .env notebooks/
   cp examples/gaia/mcp.json notebooks/
   ```

3. Set absolute paths in `.env`:
   ```bash
   GAIA_DATASET_PATH=/absolute/path/to/gaia_dataset
   ```

### Dataset Parsing Issues

**Problem:** Task data not loading correctly

**Solutions:**
1. Verify dataset structure:
   ```bash
   ls -la gaia_dataset/validation/
   # Should contain: metadata.jsonl and task files
   ```

2. Check metadata format:
   ```bash
   head -n 1 gaia_dataset/validation/metadata.jsonl | python -m json.tool
   ```

3. Validate task IDs:
   ```bash
   python -c "
   import json
   with open('gaia_dataset/validation/metadata.jsonl') as f:
       for line in f:
           task = json.loads(line)
           print(task['task_id'])
   "
   ```

### Missing Dependencies

**Problem:** ImportError for `nbformat`, `dotenv`, etc.

**Solution:**
```bash
pip install nbformat pandas python-dotenv tabulate
```

### API Key Issues

**Problem:** API calls failing or unauthorized errors

**Solutions:**
1. Verify API key is set:
   ```bash
   python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', '*' * 10 if os.getenv('LLM_API_KEY') else 'NOT SET')"
   ```

2. Test API connection:
   ```python
   from aworld.config.conf import AgentConfig
   import os
   from dotenv import load_dotenv

   load_dotenv()
   config = AgentConfig(
       llm_provider=os.getenv("LLM_PROVIDER"),
       llm_model_name=os.getenv("LLM_MODEL_NAME"),
       llm_api_key=os.getenv("LLM_API_KEY")
   )
   print(f"Provider: {config.llm_config.llm_provider}")
   print(f"Model: {config.llm_config.llm_model_name}")
   print(f"Key set: {bool(config.llm_config.llm_api_key)}")
   ```

3. Check API key format:
   - OpenAI: `sk-...`
   - Anthropic: `sk-ant-...`

### MCP Servers Not Working

**Problem:** MCP tools not available or failing

**Solutions:**
1. MCP servers are optional. The agent will run without them.

2. Install required MCP dependencies:
   ```bash
   # Google Search
   npm install -g @adenot/mcp-google-search

   # Fetch
   pip install mcp-server-fetch

   # Playwright
   npm install -g playwright-mcp-aworld
   ```

3. Verify MCP config:
   ```bash
   cat examples/gaia/mcp.json | python -m json.tool
   ```

4. Set MCP environment variables in `.env`

### Working Directory Issues

**Problem:** Files not found when running from different directories

**Solution:** The notebook automatically detects paths. Run from either:
- AWorld root: `python generate_aworld_notebook.py ...`
- Notebooks dir: `cd notebooks && jupyter notebook`

Both should work. If not, set absolute paths in the notebook.

## Advanced Usage

### Custom Dataset

To use a different dataset, modify `load_task_data()` in `generate_aworld_notebook.py`:

```python
def load_task_data_custom(dataset_path, task_id):
    # Load your custom dataset
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    # Find task
    task = next((t for t in data if t['id'] == task_id), None)

    # Return in expected format
    return {
        'task_id': task['id'],
        'Question': task['question'],
        'Final answer': task['answer'],
        'Level': task.get('difficulty', 1),
        'file_name': task.get('file', ''),
        'Annotator Metadata': {'Tools': task.get('tools', [])}
    }
```

### Custom Cell Generation

Add custom cells by creating new functions:

```python
def create_cell_custom_analysis() -> List[Any]:
    """Create custom analysis cell."""
    cells = []
    cells.append(new_markdown_cell("## Custom Analysis"))

    code = '''
# Your custom analysis code
print("Analyzing agent performance...")
'''
    cells.append(new_code_cell(code))
    return cells

# Add to generate_notebook():
cells.extend(create_cell_custom_analysis())
```

### Parallel Batch Generation

For very large batches, use parallel processing:

```bash
# Generate in chunks
for i in {0..10}; do
    start=$((i * 100))
    end=$((start + 100))
    python generate_aworld_notebook.py --start $start --end $end &
done
wait
```

## Testing Workflow

### Before Large Batch Generation

1. **Test single notebook:**
   ```bash
   python generate_aworld_notebook.py --index 0
   jupyter notebook notebooks/task_*.ipynb
   # Execute all cells and verify
   ```

2. **Test different difficulty levels:**
   ```bash
   # Level 1 (easy)
   python generate_aworld_notebook.py --index 0

   # Level 2 (medium)
   python generate_aworld_notebook.py --index 50

   # Level 3 (hard)
   python generate_aworld_notebook.py --index 100
   ```

3. **Test from different directories:**
   ```bash
   # From root
   python generate_aworld_notebook.py --index 0

   # From notebooks/
   cd notebooks && jupyter notebook
   ```

4. **Verify validation:**
   - Check answer extraction works
   - Verify ground truth comparison
   - Test with correct and incorrect answers

5. **Generate small batch:**
   ```bash
   python generate_aworld_notebook.py --start 0 --end 10
   ```

### After Generation

1. **Spot check notebooks:**
   - Open 3-5 random notebooks
   - Execute and verify all cells run
   - Check trajectory display
   - Verify tool calls shown
   - Confirm validation works

2. **Analyze results:**
   ```python
   # Count pass/fail rates
   import json
   from pathlib import Path

   notebooks = list(Path('notebooks').glob('*.ipynb'))
   print(f"Generated {len(notebooks)} notebooks")
   ```

## Git Workflow

After generating notebooks:

```bash
# Check status
git status

# Add generated files
git add notebooks/*.ipynb
git add generate_aworld_notebook.py
git add GENERATOR_README.md
git add .env.template

# Commit
git commit -m "Generate 300 notebooks for GAIA validation split

- Created notebook generator with full transparency
- Generated notebooks for tasks 0-299
- Includes trajectory display, tool calls, and validation
- 18-22 cells per notebook with markdown headers"

# Push to remote
git push origin notebooks  # or your branch name
```

## Performance Notes

- **Generation time**: ~0.1-0.5 seconds per notebook
- **Execution time**: ~10-60 seconds per notebook (depends on task complexity)
- **Notebook size**: ~10-50 KB per notebook (unexecuted)
- **With outputs**: ~50-500 KB per notebook (executed)

### Batch Generation Estimates

| Tasks | Generation Time | Execution Time (total) |
|-------|----------------|------------------------|
| 10    | ~1 second      | ~5 minutes             |
| 100   | ~10 seconds    | ~30-60 minutes         |
| 300   | ~30 seconds    | ~2-4 hours             |
| 1000  | ~2 minutes     | ~10-15 hours           |

## Examples

### Example 1: Quick Test
```bash
# Generate and run single notebook
python generate_aworld_notebook.py --index 0
jupyter notebook notebooks/task_*.ipynb
```

### Example 2: Small Batch
```bash
# Generate 20 validation tasks
python generate_aworld_notebook.py --start 0 --end 20 --split validation
```

### Example 3: Full Validation Split
```bash
# Generate all validation tasks (adjust end based on actual dataset size)
python generate_aworld_notebook.py --start 0 --end 300 --split validation

# Results in notebooks/:
# - task_1-0-0-0-0.ipynb
# - task_1-0-0-0-1.ipynb
# - task_1-0-0-1-0.ipynb
# - ... (300 total)
```

### Example 4: Specific Task IDs
```bash
# Generate specific tasks
python generate_aworld_notebook.py --id 1-0-0-0-0
python generate_aworld_notebook.py --id 2-1-3-0-5
python generate_aworld_notebook.py --id 3-2-1-4-8
```

## Support

For issues or questions:

1. Check this README troubleshooting section
2. Review the AWorld documentation
3. Check GAIA dataset documentation
4. Open an issue in the repository

## License

Same license as the AWorld project.
