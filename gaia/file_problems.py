"""Handle GAIA problems with file attachments."""

import asyncio
import shutil
import sys
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import snapshot_download

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_framework import Agent, LlmClient
from agent_tools import unzip_file, read_file, list_files, read_media_file


async def main():
    """Test GAIA problems with file attachments."""
    # Load dataset
    dataset = load_dataset("gaia-benchmark/GAIA", "2023_all", split="validation")
    
    # Download attached files
    NOTEBOOK_DIR = Path.cwd()
    PROJECT_ROOT = NOTEBOOK_DIR.parent
    CACHE_DIR = PROJECT_ROOT / "gaia_cache"
    
    snapshot_download(
        repo_id="gaia-benchmark/GAIA",
        repo_type="dataset",
        allow_patterns="2023/validation/*",
        local_dir=CACHE_DIR
    )
    
    WORK_DIR = NOTEBOOK_DIR / "gaia_workspace"
    
    def reset_workspace():
        """Restore the workspace to its initial state."""
        shutil.rmtree(WORK_DIR, ignore_errors=True)
        shutil.copytree(CACHE_DIR / "2023/validation", WORK_DIR)
        print(f"Workspace reset: {WORK_DIR}")
    
    reset_workspace()
    
    problems_with_files = [p for p in dataset if p.get('file_name')]
    problem_with_zip = [p for p in problems_with_files if p['file_name'].endswith('.zip')]
    
    print(f"Total problems: {len(dataset)}")
    print(f"Problems with attachments: {len(problems_with_files)}")
    print(f"Total problems with zip files: {len(problem_with_zip)}")
    
    problem = problem_with_zip[0]
    print(f"Question: {problem['Question'][:100]}...")
    print(f"File name: {problem['file_name']}")
    
    file_path = WORK_DIR / problem['file_name']
    print(f"File exists: {file_path.exists()}")
    
    # Reset workspace to clean state
    reset_workspace()
    
    # Select a problem with zip file attachment
    zip_problems = [p for p in problems_with_files if p['file_name'].endswith('.zip')]
    problem = zip_problems[0]
    print(problem)
    file_path = WORK_DIR / problem['file_name']
    
    # Construct prompt including file location
    prompt = f"""{problem['Question']}
 
The attached file is located at: {file_path}
"""
    print(prompt)
    
    # Create agent with file tools
    agent = Agent(
        model=LlmClient(model="gpt-5"),
        tools=[unzip_file, read_file, list_files, read_media_file],
        instructions="""You are a helpful assistant that can work with files.

CRITICAL: You MUST use the read_file tool for Excel files (.xlsx), (.xls). The tool WILL work - do NOT say you cannot read Excel files. Always call the tool first.

Tools available:
- read_file: Reads .xlsx, .xls, .csv, .txt, .py, .json, .md, .xml files. USE THIS FOR EXCEL FILES.
- read_media_file: Analyzes PDF files, images, and audio. Requires a query parameter.
- list_files: Lists directory contents
- unzip_file: Extracts zip archives


""",
        max_steps=15    )
    
    response = await agent.run(prompt)
    print(response.output)
    print(response.context)


if __name__ == "__main__":
    asyncio.run(main())
