# GitHub Repository Structure for Tutorial Series

This document outlines the recommended GitHub repository structure for organizing the tutorial series code.

---

## Repository Organization

```
ai-agent-from-scratch/
├── README.md                          # Main repository README
├── LICENSE                            # License file
├── pyproject.toml                     # Project configuration
├── requirements.txt                   # Python dependencies
│
├── agent_framework/                   # Core framework (built in episodes 3-9)
│   ├── __init__.py
│   ├── models.py                      # Episode 3
│   ├── llm.py                         # Episode 4
│   ├── agent.py                       # Episodes 5, 7
│   ├── tools.py                       # Episode 6
│   ├── mcp.py                         # Episode 8
│   ├── memory.py                      # Episode 9
│   ├── callbacks.py                   # Episode 9
│   ├── utils.py
│   └── README.md
│
├── agent_tools/                       # Built-in tools (Episode 6-7)
│   ├── __init__.py
│   ├── file_tools.py
│   ├── web_tools.py
│   ├── math_tools.py
│   └── README.md
│
├── web_app/                           # Web deployment (Episode 10)
│   ├── app.py
│   ├── static/
│   │   └── index.html
│   ├── uploads/
│   └── README.md
│
├── examples/                           # Example scripts
│   ├── example_agent.py
│   ├── test_session.py
│   └── README.md
│
├── misc/
│   └── tutorials/                     # Tutorial materials
│       ├── FEATURE_DOCUMENTATION.md
│       ├── ARCHITECTURE_DIAGRAMS.md
│       ├── GITHUB_STRUCTURE.md
│       ├── EPISODE_01_INTRODUCTION.md
│       ├── EPISODE_02_LLM_CALL.md
│       ├── EPISODE_03_DATA_MODELS.md
│       ├── EPISODE_04_LLM_CLIENT.md
│       ├── EPISODE_05_AGENT_LOOP.md
│       ├── EPISODE_06_TOOL_SYSTEM.md
│       ├── EPISODE_07_TOOL_EXECUTION.md
│       ├── EPISODE_08_MCP.md
│       ├── EPISODE_09_MEMORY.md
│       ├── EPISODE_10_WEB_DEPLOYMENT.md
│       └── exercises/
│
└── .github/
    └── workflows/                     # CI/CD (optional)
        └── tests.yml
```

---

## Branch Strategy

### Main Branch
- `main`: Complete, working codebase
- Always stable
- Production-ready

### Episode Branches
- `episode-1`: Python foundations (concepts only)
- `episode-2`: LLM client basics
- `episode-3`: Data models complete
- `episode-4`: LLM client complete
- `episode-5`: Basic agent loop
- `episode-6`: Tool system complete
- `episode-7`: Complete agent with tools
- `episode-8`: MCP integration
- `episode-9`: Memory management
- `episode-10`: Web deployment

### Feature Branches (Optional)
- `feature/session-db`: Database session manager
- `feature/streaming`: Streaming responses
- `feature/auth`: User authentication

---

## Commit Message Convention

Use clear, descriptive commit messages that match episodes:

```
Episode 3: Add Message, ToolCall, ToolResult models
Episode 3: Add Event model with timestamp
Episode 4: Implement LlmClient with LiteLLM
Episode 4: Add build_messages() function
Episode 5: Create basic Agent class
Episode 5: Implement agent.run() method
Episode 6: Add BaseTool abstract class
Episode 6: Implement FunctionTool wrapper
Episode 6: Add @tool decorator
Episode 7: Implement tool execution in agent
Episode 7: Add error handling for tools
Episode 8: Add MCP integration
Episode 9: Implement session management
Episode 9: Add memory optimization
Episode 10: Create FastAPI backend
Episode 10: Build frontend interface
```

---

## README Structure

### Main README.md

```markdown
# AI Agent Framework from Scratch

A complete AI agent framework built from scratch, designed for learning and production use.

## Features

- Multi-step reasoning with tools
- Session persistence
- Memory optimization
- MCP integration
- Web deployment

## Quick Start

\`\`\`bash
pip install -e .
python examples/example_agent.py
\`\`\`

## Tutorial Series

This repository accompanies a 10-part YouTube tutorial series:

1. [Episode 1: Introduction & Python Foundations](./misc/tutorials/EPISODE_01_INTRODUCTION.md)
2. [Episode 2: Your First LLM Call](./misc/tutorials/EPISODE_02_LLM_CALL.md)
3. [Episode 3: Core Data Models](./misc/tutorials/EPISODE_03_DATA_MODELS.md)
4. [Episode 4: The LLM Client](./misc/tutorials/EPISODE_04_LLM_CLIENT.md)
5. [Episode 5: The Basic Agent Loop](./misc/tutorials/EPISODE_05_AGENT_LOOP.md)
6. [Episode 6: Building the Tool System](./misc/tutorials/EPISODE_06_TOOL_SYSTEM.md)
7. [Episode 7: Tool Execution & Complete Agent](./misc/tutorials/EPISODE_07_TOOL_EXECUTION.md)
8. [Episode 8: MCP Integration](./misc/tutorials/EPISODE_08_MCP.md)
9. [Episode 9: Session & Memory Management](./misc/tutorials/EPISODE_09_MEMORY.md)
10. [Episode 10: Web Deployment](./misc/tutorials/EPISODE_10_WEB_DEPLOYMENT.md)

## Documentation

- [Feature Documentation](./misc/tutorials/FEATURE_DOCUMENTATION.md)
- [Architecture Diagrams](./misc/tutorials/ARCHITECTURE_DIAGRAMS.md)

## Contributing

Contributions welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.
```

---

## Tagging Strategy

Tag releases to match episodes:

```bash
# Episode milestones
git tag -a v0.1.0-episode-3 -m "Episode 3: Data Models Complete"
git tag -a v0.2.0-episode-5 -m "Episode 5: Basic Agent Loop"
git tag -a v0.3.0-episode-7 -m "Episode 7: Complete Agent"
git tag -a v1.0.0-episode-10 -m "Episode 10: Full Framework"

# Push tags
git push origin --tags
```

---

## Issue Templates

### Bug Report Template

```markdown
**Episode**: [Which episode?]
**Component**: [agent_framework/agent.py, etc.]
**Description**: [What's wrong?]
**Steps to Reproduce**: [How to reproduce]
**Expected Behavior**: [What should happen]
**Actual Behavior**: [What actually happens]
```

### Feature Request Template

```markdown
**Episode**: [Which episode?]
**Feature**: [What feature?]
**Use Case**: [Why is this needed?]
**Proposed Solution**: [How should it work?]
```

---

## Pull Request Template

```markdown
## Description
[What does this PR do?]

## Episode
[Which episode does this relate to?]

## Changes
- [ ] Added new feature
- [ ] Fixed bug
- [ ] Updated documentation
- [ ] Added tests

## Testing
[How was this tested?]

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] Episode branch updated
```

---

## GitHub Actions (Optional)

### CI Workflow

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: pip install -e .
      - run: pip install pytest
      - run: pytest tests/
```

---

## Documentation Organization

### Episode-Specific Documentation

Each episode branch should include:
- Code comments explaining decisions
- Docstrings for all functions/classes
- Inline comments for complex logic
- README in relevant directories

### Example: Episode 3 Branch

```
episode-3/
├── agent_framework/
│   ├── models.py          # Well-commented
│   └── README.md          # Explains models
└── examples/
    └── test_models.py     # Example usage
```

---

## File Naming Conventions

- **Python files**: `snake_case.py`
- **Markdown files**: `UPPER_CASE.md` for episodes, `lowercase.md` for docs
- **Directories**: `snake_case/`
- **Tests**: `test_*.py` or `*_test.py`

---

## Code Style

- Follow PEP 8
- Use type hints
- Document with docstrings
- Keep functions focused
- Use meaningful names

---

## Release Process

1. Complete episode
2. Update episode branch
3. Merge to main
4. Tag release
5. Update documentation
6. Create release notes

---

## Community Guidelines

- Be respectful
- Provide constructive feedback
- Follow episode structure
- Test before submitting
- Document your changes

---

## Resources

- [Python Style Guide (PEP 8)](https://peps.python.org/pep-0008/)
- [Semantic Versioning](https://semver.org/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

This structure ensures:
- Clear progression through episodes
- Easy navigation
- Good organization
- Professional presentation
- Community contribution support

