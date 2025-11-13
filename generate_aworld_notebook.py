#!/usr/bin/env python3
"""
AWorld MAS Notebook Generator

Generates Jupyter notebooks that demonstrate transparent, step-by-step
agent execution for tasks from the GAIA dataset.

Usage:
    python generate_aworld_notebook.py --id 1-0-0-0-0
    python generate_aworld_notebook.py --index 0
    python generate_aworld_notebook.py --start 0 --end 10 --split validation
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_task_data(
    dataset_path: str,
    split: str = "validation",
    task_id: Optional[str] = None,
    index: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    Load a single task from the GAIA dataset.

    Args:
        dataset_path: Path to GAIA dataset directory
        split: Dataset split (validation/test)
        task_id: Specific task ID to load
        index: Index in dataset to load

    Returns:
        Task data dictionary or None if not found
    """
    data_dir = Path(dataset_path) / split
    metadata_file = data_dir / "metadata.jsonl"

    if not metadata_file.exists():
        logger.error(f"Metadata file not found: {metadata_file}")
        return None

    with open(metadata_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    tasks = []
    for line in lines:
        data = json.loads(line)
        # Skip placeholder task
        if data["task_id"] == "0-0-0-0-0":
            continue
        tasks.append(data)

    # Find requested task
    if task_id:
        for task in tasks:
            if task["task_id"] == task_id:
                return task
        logger.error(f"Task ID '{task_id}' not found")
        return None
    elif index is not None:
        if 0 <= index < len(tasks):
            return tasks[index]
        else:
            logger.error(f"Index {index} out of range (0-{len(tasks)-1})")
            return None
    else:
        logger.error("Must specify either task_id or index")
        return None


def create_cell_1_task_display(task: Dict[str, Any]) -> List[Any]:
    """Create cells to display the task information."""
    cells = []

    # Markdown header
    md_content = "# AWorld MAS Task Execution\n\n"
    md_content += "This notebook demonstrates transparent, step-by-step agent execution for a GAIA benchmark task.\n"
    cells.append(new_markdown_cell(md_content))

    # Code cell to display task details
    code = f'''# Task Information
task_id = "{task['task_id']}"
level = {task['Level']}
question = """{task['Question']}"""
ground_truth = """{task['Final answer']}"""
file_name = "{task.get('file_name', '')}"
annotator_tools = {task.get('Annotator Metadata', {}).get('Tools', [])}

print("=" * 80)
print("TASK DETAILS")
print("=" * 80)
print(f"Task ID: {{task_id}}")
print(f"Difficulty Level: {{level}}")
print(f"Has File Attachment: {{bool(file_name)}}")
if file_name:
    print(f"  File: {{file_name}}")
print(f"Annotator Tools Used: {{', '.join(annotator_tools) if annotator_tools else 'None'}}")
print()
print("QUESTION:")
print("-" * 80)
print(question)
print()
print("GROUND TRUTH ANSWER:")
print("-" * 80)
print(ground_truth)
print("=" * 80)
'''
    cells.append(new_code_cell(code))

    return cells


def create_cell_2_setup() -> List[Any]:
    """Create cells for setup and configuration."""
    cells = []

    # Markdown header
    cells.append(new_markdown_cell("## Setup & Configuration\n\nInitialize the AWorld MAS framework with robust path detection."))

    # Setup code
    code = '''# Setup: Path detection and imports
import sys
import os
import json
import logging
from pathlib import Path

# Initialize variables
agent_config = None
mcp_config = {}
available_servers = []

# Current directory paths
current_dir = Path.cwd()
parent_dir = current_dir.parent

print("=" * 80)
print("ENVIRONMENT SETUP")
print("=" * 80)

# Import AWorld modules
try:
    from aworld.agents.llm_agent import Agent
    from aworld.config.conf import AgentConfig, TaskConfig
    from aworld.core.task import Task
    from aworld.runner import Runners
    print("✓ AWorld modules imported successfully")
except ImportError as e:
    print(f"✗ ERROR importing AWorld modules: {e}")
    print("  Make sure AWorld is installed: pip install aworld")
    print("  Or from GitHub: pip install git+https://github.com/inclusionAI/AWorld.git")
    raise

# Load environment variables
try:
    from dotenv import load_dotenv

    # Search for .env file in common locations
    possible_env_paths = [
        current_dir / ".env",
        parent_dir / ".env",
        Path.home() / ".env",
    ]

    env_loaded = False
    for env_path in possible_env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            print(f"✓ Loaded environment from: {env_path}")
            env_loaded = True
            break

    if not env_loaded:
        print("⚠ No .env file found, using system environment variables")

except ImportError:
    print("⚠ python-dotenv not installed, using system environment variables")

# Load MCP configuration
try:
    possible_mcp_paths = [
        current_dir / "mcp.json",
        parent_dir / "mcp.json",
        parent_dir / "examples" / "gaia" / "mcp.json",
    ]

    mcp_loaded = False
    for mcp_path in possible_mcp_paths:
        if mcp_path.exists():
            with open(mcp_path, "r", encoding="utf-8") as f:
                mcp_config = json.load(f)
                available_servers = list(mcp_config.get("mcpServers", {}).keys())
                print(f"✓ Loaded MCP config from: {mcp_path}")
                print(f"  Available MCP servers: {available_servers}")
                mcp_loaded = True
                break

    if not mcp_loaded:
        print("⚠ No mcp.json found, agent will run without MCP servers")

except Exception as e:
    print(f"⚠ Error loading MCP config: {e}")
    print("  Agent will run without MCP servers")

# Create agent configuration
try:
    agent_config = AgentConfig(
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-4o"),
        llm_base_url=os.getenv("LLM_BASE_URL"),
        llm_api_key=os.getenv("LLM_API_KEY"),
        llm_temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
    )
    print("✓ Agent configuration created")
    print(f"  Provider: {agent_config.llm_config.llm_provider}")
    print(f"  Model: {agent_config.llm_config.llm_model_name}")
    print(f"  Temperature: {agent_config.llm_config.llm_temperature}")
except Exception as e:
    print(f"✗ ERROR creating agent config: {e}")
    raise

print("=" * 80)
'''
    cells.append(new_code_cell(code))

    return cells


def create_cell_3_agent_creation(task: Dict[str, Any]) -> List[Any]:
    """Create cells for agent initialization."""
    cells = []

    cells.append(new_markdown_cell("## Agent Initialization\n\nCreate the GAIA super agent with MCP servers for tool execution."))

    code = '''# Create GAIA super agent
system_prompt = """You are a helpful AI assistant tasked with answering questions from the GAIA benchmark.

Your goal is to provide accurate, well-reasoned answers to complex questions that may require:
- Web searches and browsing
- File reading and analysis (PDF, Excel, images, code, etc.)
- Mathematical computations
- Multi-step reasoning
- Tool usage

When you have determined the final answer, provide it in this format:
<answer>your answer here</answer>

Be thorough, use available tools when needed, and show your reasoning."""

try:
    super_agent = Agent(
        conf=agent_config,
        name="gaia_super_agent",
        system_prompt=system_prompt,
        mcp_config=mcp_config,
        mcp_servers=available_servers,
        feedback_tool_result=True
    )
    print("✓ GAIA super agent created successfully")
    print(f"  Agent name: {super_agent.name}")
    print(f"  MCP servers: {super_agent.mcp_servers if super_agent.mcp_servers else 'None'}")

except Exception as e:
    print(f"✗ ERROR creating agent: {e}")
    import traceback
    traceback.print_exc()
'''
    cells.append(new_code_cell(code))

    return cells


def create_cell_4_task_execution(task: Dict[str, Any], dataset_path: str, split: str) -> List[Any]:
    """Create cells for task execution."""
    cells = []

    cells.append(new_markdown_cell("## Task Execution\n\nRun the task with the agent and capture the full execution trajectory."))

    # Prepare the question with file path if needed
    file_instruction = ""
    if task.get('file_name'):
        file_name = task['file_name']
        # Determine file type
        if any(file_name.lower().endswith(ext) for ext in ['.pdf', '.docx', '.doc', '.txt']):
            file_instruction = f' Here are the necessary document files: {file_name}'
        elif any(file_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            file_instruction = f' Here are the necessary image files: {file_name}'
        elif any(file_name.lower().endswith(ext) for ext in ['.xlsx', '.xls', '.csv']):
            file_instruction = f' Here are the necessary table files: {file_name}, for processing excel file, you can use the excel tool or write python code to process the file step-by-step and get the information.'
        elif file_name.lower().endswith('.py'):
            file_instruction = f' Here are the necessary python files: {file_name}'
        else:
            file_instruction = f' Here are the necessary files: {file_name}'

    code = f'''# Execute the task
import time

# Prepare question with file path if needed
question_with_files = question
dataset_path = "{dataset_path}"
split = "{split}"

if file_name:
    file_path = Path(dataset_path) / split / file_name
    question_with_files += "{file_instruction}"
    print(f"Task includes file attachment: {{file_path}}")
    print(f"File exists: {{file_path.exists()}}")
    print()

print("=" * 80)
print("EXECUTING TASK")
print("=" * 80)
print("Starting agent execution...")
print()

# Create and run task
start_time = time.time()
task_result = None
task_response = None

try:
    task_obj = Task(
        input=question_with_files,
        agent=super_agent,
        conf=TaskConfig()
    )

    print(f"Task created with ID: {{task_obj.id}}")
    print("Running agent...")
    print()

    # Execute task
    task_result = Runners.sync_run_task(task=task_obj)
    task_response = task_result[task_obj.id]

    end_time = time.time()
    execution_time = end_time - start_time

    print("=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    print(f"✓ Status: {{'Success' if task_response.success else 'Failed'}}")
    print(f"✓ Execution time: {{execution_time:.2f}} seconds")
    print(f"✓ Steps taken: {{len(task_response.trajectory) if task_response.trajectory else 'N/A'}}")
    if hasattr(task_response, 'usage') and task_response.usage:
        print(f"✓ Token usage: {{task_response.usage}}")
    print()
    print("AGENT ANSWER:")
    print("-" * 80)
    print(task_response.answer)
    print("=" * 80)

except Exception as e:
    print(f"✗ ERROR during task execution: {{e}}")
    import traceback
    traceback.print_exc()
'''
    cells.append(new_code_cell(code))

    return cells


def create_cell_5_trajectory_display() -> List[Any]:
    """Create cells to display execution trajectory."""
    cells = []

    cells.append(new_markdown_cell("## Execution Trajectory\n\nDetailed step-by-step breakdown of agent actions."))

    code = '''# Display execution trajectory
if task_response and hasattr(task_response, 'trajectory') and task_response.trajectory:
    print("=" * 80)
    print(f"TRAJECTORY: {len(task_response.trajectory)} STEPS")
    print("=" * 80)
    print()

    for step_idx, step in enumerate(task_response.trajectory, 1):
        print(f"{'='*80}")
        print(f"STEP {step_idx}/{len(task_response.trajectory)}")
        print(f"{'='*80}")

        # Display step information based on type
        if isinstance(step, dict):
            for key, value in step.items():
                print(f"{key}: {value}")
        else:
            print(f"Step data: {step}")

        print()
else:
    print("No trajectory data available")
    if task_response:
        print(f"Task response type: {type(task_response)}")
        print(f"Available attributes: {dir(task_response)}")
'''
    cells.append(new_code_cell(code))

    return cells


def create_cell_6_tool_calls() -> List[Any]:
    """Create cells to display MCP tool calls and results."""
    cells = []

    cells.append(new_markdown_cell("## MCP Tool Calls\n\nDetailed view of all tool executions during the task."))

    code = '''# Extract and display tool calls
if task_response and hasattr(task_response, 'trajectory') and task_response.trajectory:
    tool_calls = []

    # Extract tool calls from trajectory
    for step_idx, step in enumerate(task_response.trajectory, 1):
        if isinstance(step, dict):
            # Check for tool-related keys
            if 'tool_name' in step or 'action_name' in step:
                tool_calls.append({
                    'step': step_idx,
                    'data': step
                })

    if tool_calls:
        print("=" * 80)
        print(f"TOOL CALLS: {len(tool_calls)} total")
        print("=" * 80)
        print()

        for call in tool_calls:
            step_num = call['step']
            data = call['data']

            print(f"{'─'*80}")
            print(f"Tool Call at Step {step_num}")
            print(f"{'─'*80}")

            tool_name = data.get('tool_name', 'Unknown')
            action_name = data.get('action_name', 'Unknown')
            params = data.get('params', {})
            result = data.get('result', 'No result captured')

            print(f"Tool: {tool_name}")
            print(f"Action: {action_name}")
            print(f"\\nParameters:")
            for key, value in params.items():
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                print(f"  {key}: {value_str}")

            print(f"\\nResult:")
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            print(f"  {result_str}")
            print()
    else:
        print("No tool calls found in trajectory")
else:
    print("No trajectory available to extract tool calls")
'''
    cells.append(new_code_cell(code))

    return cells


def create_cell_7_agent_messages() -> List[Any]:
    """Create cells to display agent messages and LLM interactions."""
    cells = []

    cells.append(new_markdown_cell("## Agent Messages & LLM Interactions\n\nDetailed view of all agent communications and LLM calls."))

    code = '''# Extract and display agent messages
if task_response and hasattr(task_response, 'trajectory') and task_response.trajectory:
    print("=" * 80)
    print("AGENT MESSAGES & LLM CALLS")
    print("=" * 80)
    print()

    for step_idx, step in enumerate(task_response.trajectory, 1):
        print(f"{'─'*80}")
        print(f"Step {step_idx}: Message Details")
        print(f"{'─'*80}")

        if isinstance(step, dict):
            # Look for message-related fields
            if 'role' in step or 'content' in step or 'message' in step:
                role = step.get('role', 'unknown')
                content = step.get('content', step.get('message', ''))

                print(f"Role: {role}")
                print(f"Content:")
                content_str = str(content)
                if len(content_str) > 1000:
                    print(f"  {content_str[:1000]}...")
                    print(f"  ... ({len(content_str) - 1000} more characters)")
                else:
                    print(f"  {content_str}")
            else:
                # Display all step data
                for key, value in step.items():
                    value_str = str(value)
                    if len(value_str) > 300:
                        value_str = value_str[:300] + "..."
                    print(f"{key}: {value_str}")
        else:
            print(f"Step type: {type(step)}")
            step_str = str(step)
            if len(step_str) > 500:
                print(f"{step_str[:500]}...")
            else:
                print(step_str)

        print()
else:
    print("No trajectory available")
'''
    cells.append(new_code_cell(code))

    return cells


def create_cell_8_validation(task: Dict[str, Any]) -> List[Any]:
    """Create cells for answer validation."""
    cells = []

    cells.append(new_markdown_cell("## Answer Validation\n\nExtract the agent's answer and compare with ground truth."))

    code = '''# Extract and validate answer
import re
import string

def normalize_str(input_str, remove_punct=True):
    """Normalize string for comparison."""
    no_spaces = re.sub(r"\\s", "", input_str)
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    else:
        return no_spaces.lower()

def normalize_number_str(number_str):
    """Normalize number string."""
    for char in ["$", "%", ","]:
        number_str = number_str.replace(char, "")
    try:
        return float(number_str)
    except ValueError:
        return float("inf")

def is_float(element):
    """Check if element can be converted to float."""
    try:
        float(element)
        return True
    except (ValueError, TypeError):
        return False

def question_scorer(model_answer, ground_truth):
    """Score the model answer against ground truth."""
    try:
        if is_float(ground_truth):
            # Numeric comparison
            normalized_answer = normalize_number_str(model_answer)
            return normalized_answer == float(ground_truth)
        elif any(char in ground_truth for char in [",", ";"]):
            # List comparison
            gt_elems = re.split(r"[,;]", ground_truth)
            ma_elems = re.split(r"[,;]", model_answer)

            if len(gt_elems) != len(ma_elems):
                return False

            comparisons = []
            for ma_elem, gt_elem in zip(ma_elems, gt_elems):
                if is_float(gt_elem):
                    normalized_ma_elem = normalize_number_str(ma_elem)
                    comparisons.append(normalized_ma_elem == float(gt_elem))
                else:
                    ma_elem = normalize_str(ma_elem, remove_punct=False)
                    gt_elem = normalize_str(gt_elem, remove_punct=False)
                    comparisons.append(ma_elem == gt_elem)
            return all(comparisons)
        else:
            # String comparison
            ma_elem = normalize_str(model_answer)
            gt_elem = normalize_str(ground_truth)
            return ma_elem == gt_elem
    except Exception as e:
        print(f"Error during validation: {e}")
        return False

# Extract answer
extracted_answer = None
if task_response:
    agent_response = task_response.answer

    # Try to extract answer from <answer> tags
    match = re.search(r"<answer>(.*?)</answer>", agent_response, re.DOTALL)
    if match:
        extracted_answer = match.group(1).strip()
        print("✓ Extracted answer from <answer> tags")
    else:
        # Fallback: use full response
        extracted_answer = agent_response.strip()
        print("⚠ No <answer> tags found, using full response")

    print()
    print("=" * 80)
    print("ANSWER EXTRACTION")
    print("=" * 80)
    print("Extracted Answer:")
    print("-" * 80)
    print(extracted_answer)
    print()
    print("Ground Truth:")
    print("-" * 80)
    print(ground_truth)
    print("=" * 80)
    print()

    # Validate
    is_correct = question_scorer(extracted_answer, ground_truth)

    print("=" * 80)
    print("VALIDATION RESULT")
    print("=" * 80)
    if is_correct:
        print("✅ PASS - Answer matches ground truth!")
    else:
        print("❌ FAIL - Answer does not match ground truth")
    print("=" * 80)

    # Display comparison details
    print()
    print("Comparison Details:")
    print(f"  Task ID: {task_id}")
    print(f"  Level: {level}")
    print(f"  Correct: {is_correct}")
else:
    print("✗ No task response available for validation")
'''
    cells.append(new_code_cell(code))

    return cells


def generate_notebook(
    task: Dict[str, Any],
    output_path: Path,
    dataset_path: str,
    split: str
) -> bool:
    """
    Generate a complete notebook for the given task.

    Args:
        task: Task data dictionary
        output_path: Path to save the notebook
        dataset_path: Path to dataset directory
        split: Dataset split

    Returns:
        True if successful, False otherwise
    """
    try:
        cells = []

        # Generate all cells
        cells.extend(create_cell_1_task_display(task))
        cells.extend(create_cell_2_setup())
        cells.extend(create_cell_3_agent_creation(task))
        cells.extend(create_cell_4_task_execution(task, dataset_path, split))
        cells.extend(create_cell_5_trajectory_display())
        cells.extend(create_cell_6_tool_calls())
        cells.extend(create_cell_7_agent_messages())
        cells.extend(create_cell_8_validation(task))

        # Create notebook
        nb = new_notebook(cells=cells)

        # Write to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)

        logger.info(f"✓ Generated notebook: {output_path}")
        return True

    except Exception as e:
        logger.error(f"✗ Error generating notebook: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate AWorld MAS notebooks for GAIA dataset tasks"
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Task ID to generate notebook for (e.g., 1-0-0-0-0)"
    )
    parser.add_argument(
        "--index",
        type=int,
        help="Dataset index to generate notebook for (0-based)"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for batch generation"
    )
    parser.add_argument(
        "--end",
        type=int,
        help="End index for batch generation"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["validation", "test"],
        help="Dataset split to use"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to GAIA dataset directory (default: from GAIA_DATASET_PATH env var or ./gaia_dataset)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="notebooks",
        help="Output directory for notebooks"
    )

    args = parser.parse_args()

    # Get dataset path
    dataset_path = args.dataset_path or os.getenv("GAIA_DATASET_PATH", "./gaia_dataset")
    dataset_path = str(Path(dataset_path).resolve())

    logger.info(f"Dataset path: {dataset_path}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Output directory: {args.output_dir}")

    # Single task generation
    if args.id or args.index is not None:
        task = load_task_data(
            dataset_path=dataset_path,
            split=args.split,
            task_id=args.id,
            index=args.index
        )

        if task is None:
            logger.error("Failed to load task")
            return 1

        # Generate output filename
        task_id_safe = task['task_id'].replace('/', '_')
        output_path = Path(args.output_dir) / f"task_{task_id_safe}.ipynb"

        logger.info(f"Generating notebook for task: {task['task_id']}")
        success = generate_notebook(task, output_path, dataset_path, args.split)

        return 0 if success else 1

    # Batch generation
    elif args.end is not None:
        logger.info(f"Batch generation: indices {args.start} to {args.end}")

        successes = 0
        failures = 0

        for idx in range(args.start, args.end):
            try:
                task = load_task_data(
                    dataset_path=dataset_path,
                    split=args.split,
                    index=idx
                )

                if task is None:
                    logger.warning(f"Skipping index {idx}: task not found")
                    failures += 1
                    continue

                task_id_safe = task['task_id'].replace('/', '_')
                output_path = Path(args.output_dir) / f"task_{task_id_safe}.ipynb"

                if generate_notebook(task, output_path, dataset_path, args.split):
                    successes += 1
                else:
                    failures += 1

            except Exception as e:
                logger.error(f"Error processing index {idx}: {e}")
                failures += 1
                continue

        logger.info(f"\nBatch generation complete:")
        logger.info(f"  ✓ Successes: {successes}")
        logger.info(f"  ✗ Failures: {failures}")

        return 0 if failures == 0 else 1

    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
