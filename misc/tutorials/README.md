# Tutorial Series Materials

This directory contains all materials for the "Building an AI Agent Framework from Scratch" YouTube tutorial series.

---

## 📚 Documentation

### Core Documentation
- **[FEATURE_DOCUMENTATION.md](./FEATURE_DOCUMENTATION.md)**: Complete inventory of all framework features
- **[ARCHITECTURE_DIAGRAMS.md](./ARCHITECTURE_DIAGRAMS.md)**: Visual diagrams using Mermaid syntax
- **[GITHUB_STRUCTURE.md](./GITHUB_STRUCTURE.md)**: Repository organization and branch strategy
- **[EXERCISES.md](./EXERCISES.md)**: Exercises and challenges for each episode

---

## 🎬 Episode Guides

### Episode 1: Introduction & Python Foundations
**[EPISODE_01_INTRODUCTION.md](./EPISODE_01_INTRODUCTION.md)**
- Python patterns: Pydantic, dataclasses, async/await
- Type hints and validation
- Duration: 30 minutes

### Episode 2: Your First LLM Call
**[EPISODE_02_LLM_CALL.md](./EPISODE_02_LLM_CALL.md)**
- Chat completion API format
- LiteLLM integration
- Error handling
- Duration: 25 minutes

### Episode 3: Core Data Models
**[EPISODE_03_DATA_MODELS.md](./EPISODE_03_DATA_MODELS.md)**
- Message, ToolCall, ToolResult models
- Event and ExecutionContext
- Pydantic vs Dataclass
- Duration: 30 minutes

### Episode 4: The LLM Client
**[EPISODE_04_LLM_CLIENT.md](./EPISODE_04_LLM_CLIENT.md)**
- LlmRequest and LlmResponse models
- build_messages() function
- Response parsing
- Duration: 30 minutes

### Episode 5: The Basic Agent Loop
**[EPISODE_05_AGENT_LOOP.md](./EPISODE_05_AGENT_LOOP.md)**
- Think-Act-Observe cycle
- Agent.run() and Agent.step()
- Execution tracking
- Duration: 35 minutes

### Episode 6: Building the Tool System
**[EPISODE_06_TOOL_SYSTEM.md](./EPISODE_06_TOOL_SYSTEM.md)**
- BaseTool abstract class
- FunctionTool wrapper
- @tool decorator
- Schema generation
- Duration: 35 minutes

### Episode 7: Tool Execution & Complete Agent
**[EPISODE_07_TOOL_EXECUTION.md](./EPISODE_07_TOOL_EXECUTION.md)**
- Tool execution in agent loop
- Error handling
- Complete working agent
- Duration: 35 minutes

### Episode 8: MCP Integration
**[EPISODE_08_MCP.md](./EPISODE_08_MCP.md)**
- Model Context Protocol
- Tool discovery
- MCP server integration
- Duration: 30 minutes

### Episode 9: Session & Memory Management
**[EPISODE_09_MEMORY.md](./EPISODE_09_MEMORY.md)**
- Session persistence
- Token counting
- Memory optimization strategies
- Duration: 35 minutes

### Episode 10: Web Deployment
**[EPISODE_10_WEB_DEPLOYMENT.md](./EPISODE_10_WEB_DEPLOYMENT.md)**
- FastAPI backend
- Frontend interface
- File uploads
- Session management
- Duration: 35 minutes

---

## 📊 Series Overview

**Total Duration**: ~5.5 hours  
**Target Audience**: Intermediate Python developers  
**Teaching Style**: Build from scratch (live coding)  
**Prerequisites**: Python 3.10+, basic async knowledge

---

## 🎯 Learning Path

```
Episode 1-2: Foundations
    ↓
Episode 3-4: Core Components
    ↓
Episode 5-7: Agent System
    ↓
Episode 8-9: Advanced Features
    ↓
Episode 10: Deployment
```

---

## 📁 File Structure

```
misc/tutorials/
├── README.md (this file)
├── FEATURE_DOCUMENTATION.md
├── ARCHITECTURE_DIAGRAMS.md
├── GITHUB_STRUCTURE.md
├── EXERCISES.md
├── EPISODE_01_INTRODUCTION.md
├── EPISODE_02_LLM_CALL.md
├── EPISODE_03_DATA_MODELS.md
├── EPISODE_04_LLM_CLIENT.md
├── EPISODE_05_AGENT_LOOP.md
├── EPISODE_06_TOOL_SYSTEM.md
├── EPISODE_07_TOOL_EXECUTION.md
├── EPISODE_08_MCP.md
├── EPISODE_09_MEMORY.md
└── EPISODE_10_WEB_DEPLOYMENT.md
```

---

## 🚀 Quick Start

1. **Read the Feature Documentation** to understand what we're building
2. **Follow episodes in order** - each builds on the previous
3. **Complete exercises** after each episode
4. **Check GitHub branches** for episode-specific code

---

## 💡 Tips for Teaching

### For Each Episode:
1. **Hook** (2 min): Show what we'll build
2. **Problem** (3 min): Why do we need this?
3. **Concept** (5 min): How does it work?
4. **Live Coding** (15-20 min): Build it step by step
5. **Demo** (3 min): Show it working
6. **Next Steps** (2 min): Preview next episode

### Visual Aids:
- Use architecture diagrams from `ARCHITECTURE_DIAGRAMS.md`
- Show code side-by-side with explanations
- Use terminal output to show execution
- Display execution traces

### Engagement:
- Ask rhetorical questions
- Show "what if" scenarios
- Compare with alternatives
- Highlight design decisions
- Show common mistakes

---

## 📝 Episode Checklist

Before recording each episode:

- [ ] Review episode outline
- [ ] Prepare code examples
- [ ] Set up development environment
- [ ] Test all code snippets
- [ ] Prepare visual aids
- [ ] Review architecture diagrams
- [ ] Prepare exercises
- [ ] Check GitHub branch

---

## 🎬 Recording Tips

1. **Start fresh**: Begin each episode with clean files
2. **Build incrementally**: Test after each major component
3. **Show errors**: Demonstrate common mistakes and fixes
4. **Explain decisions**: Why this approach vs alternatives
5. **Keep it real**: Show actual debugging process
6. **Engage audience**: Ask questions, pause for thought

---

## 📚 Additional Resources

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [LiteLLM Documentation](https://docs.litellm.ai/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MCP Specification](https://modelcontextprotocol.io/)

---

## 🤝 Contributing

Found an error or want to improve the tutorials?

1. Fork the repository
2. Make your changes
3. Submit a pull request
4. Include explanation of changes

---

## 📧 Support

Questions or issues?
- Open a GitHub issue
- Check existing documentation
- Review episode-specific guides

---

## 🎉 Series Completion

After completing all 10 episodes, you will have:

✅ Built a complete AI agent framework  
✅ Understand every component  
✅ Created production-ready code  
✅ Deployed a web application  
✅ Gained deep understanding of agent architecture

**Congratulations on your learning journey!**

---

*Last Updated: 2026*

