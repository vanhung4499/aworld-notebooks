# AWorld Notebook Generator 📓

Generate transparent, executable Jupyter notebooks for testing AI agents on the [GAIA benchmark](https://huggingface.co/gaia-benchmark). Each notebook demonstrates step-by-step agent execution with full visibility into tool calls, reasoning, and validation.

## ✨ Features

- 🔍 **Full Transparency**: Every agent action, tool call, and LLM interaction is visible
- 🎯 **Auto Validation**: Automatic answer extraction and comparison with ground truth
- 🛠️ **MCP Integration**: Visualize tool usage from MCP servers
- 📊 **Batch Generation**: Generate hundreds of notebooks efficiently
- 🔄 **Reproducible**: Anyone can re-run notebooks to verify agent performance

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Core dependencies
pip install nbformat pandas pyarrow python-dotenv

# Install AWorld framework
pip install aworld
# Or from source:
# pip install git+https://github.com/inclusionAI/AWorld.git
```

### 2. Download GAIA Dataset

**Recommended Method: Python + HuggingFace Hub**

```bash
pip install huggingface-hub

python3 << 'EOF'
from huggingface_hub import snapshot_download

print("Downloading GAIA dataset...")
data_dir = snapshot_download(
    repo_id="gaia-benchmark/GAIA",
    repo_type="dataset",
    local_dir="./gaia_dataset",
    local_dir_use_symlinks=False
)
print(f"✓ Downloaded to: {data_dir}")
EOF
```

**Alternative: Git with LFS**

```bash
# Install Git LFS first
brew install git-lfs  # macOS
# sudo apt-get install git-lfs  # Linux

git lfs install
git clone https://huggingface.co/datasets/gaia-benchmark/GAIA gaia_dataset
cd gaia_dataset
git lfs pull
```

### 3. Configure Environment

```bash
# Create .env file
cat > .env << 'EOF'
# LLM Configuration
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o
LLM_API_KEY=your_api_key_here
LLM_TEMPERATURE=0.0

# Dataset Path
GAIA_DATASET_PATH=./gaia_dataset/2023
EOF
```

### 4. Generate Your First Notebook

```bash
# Generate by index
python generate_aworld_notebook.py --index 0 --split validation

# Output: notebooks/task_<task-id>.ipynb
```

### 5. Run the Notebook

```bash
# Open in Jupyter
jupyter notebook notebooks/

# Or in VS Code
code notebooks/
```

## 📖 Usage Examples

### Single Notebook Generation

```bash
# By task ID
python generate_aworld_notebook.py \
    --id 04a04a9b-226c-43fd-b319-d5e89743676f \
    --split validation

# By index (0-based)
python generate_aworld_notebook.py --index 0

# Custom output directory
python generate_aworld_notebook.py \
    --index 0 \
    --output-dir my_notebooks \
    --dataset-path ./gaia_dataset/2023
```

### Batch Generation

```bash
# Generate first 10 tasks
python generate_aworld_notebook.py --start 0 --end 10

# Generate all validation tasks (~165 tasks)
python generate_aworld_notebook.py \
    --start 0 \
    --end 165 \
    --split validation

# Generate test split
python generate_aworld_notebook.py \
    --start 0 \
    --end 50 \
    --split test
```

### Using Environment Variables

```bash
# Set dataset path globally
export GAIA_DATASET_PATH="./gaia_dataset/2023"

# Now you can omit --dataset-path
python generate_aworld_notebook.py --index 0
```

## 📁 Project Structure

```
aworld-notebooks/
├── generate_aworld_notebook.py    # Main generator script
├── README.md                      # This file
├── GENERATOR_README.md            # Detailed documentation
├── QUICKSTART_NOTEBOOKS.md        # Quick reference guide
├── .env                           # Your configuration (create this)
├── .gitignore                     # Git ignore rules
│
├── gaia_dataset/                  # GAIA dataset (download separately)
│   └── 2023/
│       ├── validation/
│       │   ├── metadata.parquet   # Task definitions
│       │   ├── *.xlsx             # Attached files
│       │   ├── *.pdf
│       │   └── ...
│       └── test/
│           └── ...
│
└── notebooks/                     # Generated notebooks
    ├── task_<id>.ipynb
    └── ...
```

## 📊 Notebook Structure

Each generated notebook contains **8 sections**:

| Section | Description |
|---------|-------------|
| 1️⃣ **Task Display** | Question, ground truth, difficulty level, annotator metadata |
| 2️⃣ **Setup & Configuration** | Environment detection, imports, config loading |
| 3️⃣ **Agent Initialization** | Create agent with system prompt and MCP servers |
| 4️⃣ **Task Execution** | Run task and capture execution trajectory |
| 5️⃣ **Trajectory Display** | Step-by-step breakdown of agent actions |
| 6️⃣ **Tool Calls** | Detailed view of all MCP tool executions |
| 7️⃣ **Agent Messages** | LLM interactions and communications |
| 8️⃣ **Answer Validation** | Extract answer and compare with ground truth (✅/❌) |

## 🔧 Configuration

### Required Environment Variables

```bash
# LLM Provider (openai, anthropic, etc.)
LLM_PROVIDER=openai
LLM_MODEL_NAME=gpt-4o
LLM_API_KEY=your_api_key_here

# Dataset location
GAIA_DATASET_PATH=./gaia_dataset/2023
```

### Optional: MCP Server Configuration

Create `mcp.json` to enable additional tools:

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/allowed/path"]
    },
    "brave-search": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-brave-search"],
      "env": {
        "BRAVE_API_KEY": "your_brave_api_key"
      }
    }
  }
}
```

## 🐛 Troubleshooting

### Dataset Files Show "Git LFS pointer" Error

**Problem:** Parquet files are Git LFS pointers, not actual files.

**Solution:**
```bash
# Method 1: Pull LFS files
cd gaia_dataset
git lfs pull

# Method 2: Download via Python (recommended)
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('gaia-benchmark/GAIA', repo_type='dataset', local_dir='./gaia_dataset_full', local_dir_use_symlinks=False)"
```

### "AWorld modules not found"

**Solution:**
```bash
pip install aworld
# Or from GitHub:
pip install git+https://github.com/inclusionAI/AWorld.git
```

### "No metadata.parquet found"

**Problem:** Wrong dataset path structure.

**Solution:**
```bash
# Dataset structure should be:
# gaia_dataset/2023/validation/metadata.parquet
# NOT: gaia_dataset/validation/metadata.parquet

# Verify structure:
ls -la gaia_dataset/2023/validation/

# Use correct path:
python generate_aworld_notebook.py \
    --index 0 \
    --dataset-path "./gaia_dataset/2023"
```

### Git Push Failed (Git LFS error)

**Solution:**
```bash
# Install Git LFS
brew install git-lfs  # macOS
git lfs install

# Or ignore dataset in git:
echo -e "\ngaia_dataset/\nGAIA/" >> .gitignore
git rm -r --cached gaia_dataset/
```

## 📚 Additional Resources

- **[GENERATOR_README.md](GENERATOR_README.md)** - Comprehensive documentation
- **[QUICKSTART_NOTEBOOKS.md](QUICKSTART_NOTEBOOKS.md)** - Quick reference guide
- **[GAIA Benchmark](https://huggingface.co/gaia-benchmark)** - Official dataset page
- **[AWorld Framework](https://github.com/inclusionAI/AWorld)** - Agent framework repository

## 🤝 Contributing

Issues and pull requests are welcome! Please ensure:
- Generated notebooks run without errors
- Code follows existing style conventions
- New features are documented

## 📝 License

This project is part of the AWorld framework. See the main AWorld repository for license details.

## 🙏 Acknowledgments

- **GAIA Benchmark Team** for the comprehensive agent evaluation dataset
- **AWorld Framework** for the multi-agent system foundation

---

**Need help?** Check [GENERATOR_README.md](GENERATOR_README.md) for detailed documentation or open an issue on GitHub.

