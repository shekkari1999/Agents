# Next Steps: Complementary Skills & Learning Path

After building this agent framework from scratch, here's what to learn next to become a complete Agentic AI Engineer.

---

## Why This Matters

You've built the fundamentals. But in the real world:
- Agents need to retrieve knowledge (RAG)
- Complex tasks need multiple agents
- Production systems need observability
- Safety is non-negotiable

This guide helps you add value beyond "I built an agent framework."

---

## Priority 1: RAG (Retrieval Augmented Generation)

You have basic embeddings, but production RAG is much deeper.

### Key Concepts

| Concept | Description | Why It Matters |
|---------|-------------|----------------|
| **Chunking Strategies** | Fixed-size, semantic, recursive splitting | Affects retrieval quality dramatically |
| **Hybrid Search** | Combine vector + keyword (BM25) | Better results than vector-only |
| **Re-ranking** | Cross-encoders to improve top-k | Fixes retriever mistakes |
| **Vector Databases** | Pinecone, Weaviate, Qdrant, Chroma | Each has different tradeoffs |
| **Query Transformation** | HyDE, step-back, multi-query | Improve query-document matching |
| **Agentic RAG** | Agent decides when/what to retrieve | Most flexible approach |

### Add to Your Project

```python
@tool
def rag_search(query: str, top_k: int = 5) -> str:
    """Search knowledge base with hybrid retrieval."""
    # 1. Vector search
    vector_results = vector_db.search(embed(query), top_k=top_k*2)
    
    # 2. Keyword search (BM25)
    keyword_results = bm25_search(query, top_k=top_k*2)
    
    # 3. Merge and dedupe
    combined = merge_results(vector_results, keyword_results)
    
    # 4. Re-rank with cross-encoder
    reranked = cross_encoder.rerank(query, combined, top_k=top_k)
    
    return format_results(reranked)
```

### Resources
- [LlamaIndex](https://docs.llamaindex.ai/) - Best RAG framework
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- Paper: "Retrieval-Augmented Generation for Large Language Models: A Survey"

---

## Priority 2: Multi-Agent Systems

Your framework is single-agent. The industry is moving to multi-agent architectures.

### Patterns

| Pattern | Description | Use Case |
|---------|-------------|----------|
| **Supervisor** | One agent delegates to specialists | Complex tasks with clear subtasks |
| **Debate** | Agents argue, synthesize best answer | Reduce hallucination, improve reasoning |
| **Pipeline** | Agent A -> Agent B -> Agent C | Sequential processing |
| **Swarm** | Agents coordinate dynamically | Open-ended exploration |
| **Reflection** | Agent critiques own output | Self-improvement loop |

### Example: Supervisor Pattern

```python
class SupervisorAgent(Agent):
    def __init__(self, specialists: List[Agent]):
        self.specialists = {agent.name: agent for agent in specialists}
        super().__init__(
            instructions="""You are a supervisor. 
            Delegate tasks to specialists:
            - researcher: for information gathering
            - coder: for code tasks
            - writer: for content creation
            """
        )
    
    async def delegate(self, task: str, specialist_name: str):
        specialist = self.specialists[specialist_name]
        return await specialist.run(task)
```

### Frameworks to Study
- **LangGraph** - Stateful multi-agent workflows
- **CrewAI** - Role-based agent teams
- **AutoGen** - Microsoft's multi-agent framework
- **Swarm** - OpenAI's experimental framework

---

## Priority 3: Observability & Tracing

You have `format_trace`, but production systems need more.

### Tools

| Tool | Type | Best For |
|------|------|----------|
| **LangSmith** | SaaS | LangChain users, enterprise |
| **LangFuse** | Open Source | Self-hosted, privacy-focused |
| **Weights & Biases** | SaaS | Experiment tracking |
| **OpenTelemetry** | Standard | Distributed tracing |
| **Arize Phoenix** | Open Source | LLM observability |

### Key Metrics to Track

```python
@dataclass
class AgentMetrics:
    # Latency
    total_duration_ms: float
    llm_call_duration_ms: float
    tool_execution_duration_ms: float
    
    # Token Usage
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    
    # Cost
    estimated_cost_usd: float
    
    # Quality
    steps_to_completion: int
    tool_calls_count: int
    errors_count: int
```

### Add to Your Project

```python
# In agent.py
class Agent:
    async def run(self, ...):
        start_time = time.time()
        
        try:
            result = await self._run_internal(...)
            
            # Log metrics
            self.log_metrics(AgentMetrics(
                total_duration_ms=(time.time() - start_time) * 1000,
                steps_to_completion=result.context.current_step,
                # ... other metrics
            ))
            
            return result
        except Exception as e:
            self.log_error(e)
            raise
```

---

## Priority 4: Evaluation & Benchmarking

You have GAIA. Go deeper with systematic evaluation.

### Evaluation Types

| Type | What It Measures | How |
|------|------------------|-----|
| **Task Completion** | Did agent solve the problem? | Binary success/fail |
| **Accuracy** | Is the answer correct? | Compare to ground truth |
| **Faithfulness** | Is answer grounded in retrieved context? | LLM-as-Judge |
| **Relevance** | Is answer relevant to question? | LLM-as-Judge |
| **Latency** | How fast is the agent? | Time measurement |
| **Cost** | How much did it cost? | Token tracking |

### LLM-as-Judge Pattern

```python
JUDGE_PROMPT = """
You are evaluating an AI agent's response.

Question: {question}
Agent's Answer: {answer}
Ground Truth: {ground_truth}

Rate the answer on a scale of 1-5:
1 = Completely wrong
2 = Partially wrong
3 = Partially correct
4 = Mostly correct
5 = Completely correct

Provide your rating and reasoning.
"""

async def evaluate_with_llm(question: str, answer: str, ground_truth: str) -> int:
    response = await llm.generate(JUDGE_PROMPT.format(...))
    return extract_rating(response)
```

### Frameworks
- **Ragas** - RAG evaluation
- **DeepEval** - LLM evaluation framework
- **Promptfoo** - Prompt testing
- **Evalica** - Comparative evaluation

---

## Priority 5: Safety & Guardrails

Production agents need safety layers.

### Input Guardrails

```python
class InputGuardrails:
    def __init__(self):
        self.blocked_patterns = [
            r"ignore previous instructions",
            r"you are now",
            r"pretend to be",
        ]
    
    def check(self, input: str) -> bool:
        for pattern in self.blocked_patterns:
            if re.search(pattern, input, re.IGNORECASE):
                return False
        return True
```

### Output Guardrails

```python
class OutputGuardrails:
    async def check(self, output: str) -> tuple[bool, str]:
        # Check for PII
        if self.contains_pii(output):
            return False, "Response contains PII"
        
        # Check for harmful content
        if await self.is_harmful(output):
            return False, "Response contains harmful content"
        
        return True, ""
```

### Integration with Your Framework

```python
# Add as callbacks
agent = Agent(
    model=LlmClient(model="gpt-4o-mini"),
    tools=[...],
    before_llm_callback=input_guardrails.check,
    after_llm_callback=output_guardrails.check,
)
```

### Tools
- **Guardrails AI** - Structured output validation
- **NeMo Guardrails** - NVIDIA's safety framework
- **Lakera Guard** - Prompt injection detection
- **Rebuff** - Self-hardening prompt injection detector

---

## Priority 6: LLM Routing & Optimization

### Smart Model Selection

```python
class ModelRouter:
    def __init__(self):
        self.models = {
            "simple": "gpt-4o-mini",      # Fast, cheap
            "complex": "gpt-4o",           # Powerful
            "coding": "claude-sonnet-4-5",    # Best for code
        }
    
    async def route(self, query: str) -> str:
        # Classify query complexity
        complexity = await self.classify_complexity(query)
        
        if "code" in query.lower():
            return self.models["coding"]
        elif complexity == "high":
            return self.models["complex"]
        else:
            return self.models["simple"]
```

### Semantic Caching

```python
class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95):
        self.cache = {}
        self.embeddings = {}
        self.threshold = similarity_threshold
    
    async def get(self, query: str) -> str | None:
        query_embedding = embed(query)
        
        for cached_query, cached_response in self.cache.items():
            similarity = cosine_similarity(
                query_embedding, 
                self.embeddings[cached_query]
            )
            if similarity > self.threshold:
                return cached_response
        
        return None
    
    async def set(self, query: str, response: str):
        self.cache[query] = response
        self.embeddings[query] = embed(query)
```

---

## Suggested Learning Path

### Month 1: RAG Deep Dive
- [ ] Implement hybrid search (vector + BM25)
- [ ] Add re-ranking with cross-encoder
- [ ] Build RAGTool for your agent
- [ ] Experiment with different chunking strategies

### Month 2: Multi-Agent Systems
- [ ] Study LangGraph architecture
- [ ] Implement supervisor pattern
- [ ] Build debate/reflection agents
- [ ] Add multi-agent orchestration layer

### Month 3: Production Readiness
- [ ] Integrate LangFuse for observability
- [ ] Implement input/output guardrails
- [ ] Build evaluation suite with LLM-as-Judge
- [ ] Add cost tracking and alerts

### Month 4: Advanced Topics
- [ ] Implement smart model routing
- [ ] Add semantic caching
- [ ] Experiment with fine-tuning
- [ ] Build monitoring dashboard

---

## Quick Wins to Add Now

These can be added to your framework in a few hours each:

### 1. Semantic Caching
```python
# In memory.py
class SemanticCache:
    """Cache responses for similar queries."""
    ...
```

### 2. Cost Tracker
```python
# In agent.py
PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # per 1M tokens
    "gpt-4o": {"input": 2.50, "output": 10.00},
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    prices = PRICING.get(model, {"input": 0, "output": 0})
    return (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
```

### 3. Streaming Support
```python
# In llm.py
async def generate_streaming(self, request: LlmRequest):
    """Stream tokens as they're generated."""
    ...
```

### 4. Simple Guardrails
```python
# In callbacks.py
def prompt_injection_detector(context, request):
    """Block obvious prompt injection attempts."""
    ...
```

### 5. Retry with Exponential Backoff
```python
# In llm.py
async def generate_with_retry(self, request: LlmRequest, max_retries: int = 3):
    """Retry failed LLM calls with exponential backoff."""
    ...
```

---

## Resources

### Courses
- [DeepLearning.AI - Building Agentic RAG with LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex/)
- [DeepLearning.AI - Multi AI Agent Systems with crewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/)
- [LangChain Academy](https://academy.langchain.com/)

### Papers
- "ReAct: Synergizing Reasoning and Acting in Language Models"
- "Toolformer: Language Models Can Teach Themselves to Use Tools"
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
- "Retrieval-Augmented Generation for Large Language Models: A Survey"

### Communities
- [LangChain Discord](https://discord.gg/langchain)
- [LlamaIndex Discord](https://discord.gg/llamaindex)
- [Latent Space Podcast](https://www.latent.space/)
- [AI Engineer Newsletter](https://www.aiengineer.dev/)

---

## What Would Make Your Project Stand Out

1. **RAG + Agents** - Agent that retrieves, reasons, and acts
2. **Multi-Agent Orchestration** - Coordinator + specialists
3. **Built-in Evaluation** - Self-testing agent framework
4. **Safety Layer** - Production-grade guardrails
5. **Observability Dashboard** - Visual trace explorer
6. **Semantic Caching** - Cost optimization
7. **Model Routing** - Smart model selection

---

**Previous**: [Resume Guide](./RESUME_GUIDE.md)  
**Back to**: [Tutorial Overview](./README.md)

