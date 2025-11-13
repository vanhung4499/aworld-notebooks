# Quick Start: AWorld Notebook Generator

## 🚀 Generate Your First Notebook in 3 Steps

### Step 1: Install Dependencies
```bash
pip install nbformat pandas python-dotenv aworld
```

### Step 2: Configure Environment
```bash
# Copy template
cp .env.template .env

# Edit and add your API key
nano .env
# Set: LLM_API_KEY=your_api_key_here
```

### Step 3: Generate a Notebook
```bash
# Generate by index (uses test data)
python generate_aworld_notebook.py --index 0

# Or with real GAIA dataset (after downloading)
python generate_aworld_notebook.py --id 1-0-0-0-0 --dataset-path /path/to/gaia_dataset
```

## 📊 What You Get

Each notebook contains **16 cells** organized into **8 sections**:

1. **Task Display** - Question, ground truth, metadata
2. **Setup & Configuration** - Robust path detection, imports, config loading
3. **Agent Initialization** - Create GAIA super agent with MCP servers
4. **Task Execution** - Live agent execution with timing
5. **Execution Trajectory** - Step-by-step breakdown
6. **MCP Tool Calls** - Detailed tool execution logs
7. **Agent Messages** - LLM interactions and communications
8. **Answer Validation** - Extract answer, compare with ground truth (✅/❌)

## 🎯 Common Use Cases

### Generate Single Notebook
```bash
# By task ID
./generate_aworld_notebook.py --id 1-0-0-0-0

# By index
./generate_aworld_notebook.py --index 0
```

### Generate Batch
```bash
# 10 notebooks
./generate_aworld_notebook.py --start 0 --end 10

# Full validation split (~300 tasks)
./generate_aworld_notebook.py --start 0 --end 300 --split validation
```

### Run Generated Notebook
```bash
# Start Jupyter
jupyter notebook notebooks/

# Or use VS Code
code notebooks/task_1-0-0-0-0.ipynb
```

## 📁 Directory Structure

```
AWorld/
├── generate_aworld_notebook.py  # Main generator script
├── GENERATOR_README.md           # Full documentation
├── QUICKSTART_NOTEBOOKS.md       # This file
├── .env.template                 # Environment template
├── .env                          # Your config (create this)
├── gaia_dataset/                 # GAIA dataset (download separately)
│   └── validation/
│       ├── metadata.jsonl
│       └── files/
└── notebooks/                    # Generated notebooks (auto-created)
    ├── task_1-0-0-0-0.ipynb
    ├── task_1-0-0-0-1.ipynb
    └── ...
```

## 🔧 Troubleshooting

### "ModuleNotFoundError: No module named 'aworld'"
```bash
# Install AWorld in editable mode
pip install -e .
```

### "API Key not set"
```bash
# Check your .env file
cat .env | grep LLM_API_KEY

# Make sure it's set
echo "LLM_API_KEY=sk-your-key-here" >> .env
```

### "GAIA dataset not found"
The generator includes test data for quick testing. For real GAIA tasks:

```bash
# Download from HuggingFace
git lfs install
git clone https://huggingface.co/datasets/gaia-benchmark/GAIA gaia_dataset
```

## 📖 Need More Help?

- See **GENERATOR_README.md** for comprehensive documentation
- Check examples in `notebooks/` directory
- Test with: `./generate_aworld_notebook.py --help`

## ✅ Verification

After generating a notebook, verify it works:

```bash
# 1. Generate test notebook
./generate_aworld_notebook.py --index 0

# 2. Check it was created
ls -lh notebooks/task_*.ipynb

# 3. Verify structure
python -c "import nbformat; nb = nbformat.read('notebooks/task_1-0-0-0-0.ipynb', as_version=4); print(f'✓ Notebook has {len(nb.cells)} cells')"

# 4. Open and run in Jupyter
jupyter notebook notebooks/task_1-0-0-0-0.ipynb
```

Expected output: `✓ Notebook has 16 cells`

## 🎓 Example Workflow

Complete workflow from setup to execution:

```bash
# 1. Setup (one-time)
pip install nbformat pandas python-dotenv
cp .env.template .env
# Edit .env and add API key

# 2. Generate notebooks
./generate_aworld_notebook.py --start 0 --end 10

# 3. Run notebooks
jupyter notebook notebooks/

# 4. (In Jupyter) Open a notebook and click "Run All"

# 5. See results with validation at the bottom (✅/❌)
```

## 🚀 Pro Tips

1. **Test first**: Generate 1-2 notebooks before large batches
2. **Check API costs**: Running 300 notebooks can use significant API credits
3. **Use specific tasks**: Target interesting tasks with `--id` for demos
4. **Parallel generation**: Notebooks generate fast (~0.1s each), execution is slower
5. **Version control**: Add notebooks to git for reproducibility

## 📈 Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Generate 1 notebook | ~0.1s | Creation only |
| Execute 1 notebook | 10-60s | Depends on task complexity |
| Generate 100 notebooks | ~10s | No execution |
| Execute 100 notebooks | 30-60min | With API calls |

## 🎉 Success!

If you see output like this, you're good to go:

```
2025-11-04 18:15:00,075 - INFO - Batch generation: indices 0 to 3
2025-11-04 18:15:00,142 - INFO - ✓ Generated notebook: notebooks/task_1-0-0-0-0.ipynb
2025-11-04 18:15:00,153 - INFO - ✓ Generated notebook: notebooks/task_1-0-0-0-1.ipynb
2025-11-04 18:15:00,162 - INFO - ✓ Generated notebook: notebooks/task_2-0-0-1-0.ipynb
2025-11-04 18:15:00,162 - INFO - Batch generation complete:
2025-11-04 18:15:00,162 - INFO -   ✓ Successes: 3
2025-11-04 18:15:00,162 - INFO -   ✗ Failures: 0
```

Happy notebook generating! 🎊
