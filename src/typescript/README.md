# Multi-Agent Orchestration Patterns (TypeScript)

This module demonstrates four fundamental patterns for orchestrating multiple LLM agents:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION PATTERNS                               │
│                                                                         │
│  1. SEQUENTIAL CHAIN                2. PARALLEL FAN-OUT                 │
│     Input → A → B → C → Output         Input ──┬→ A ─┐                  │
│                                                ├→ B ─┼→ Aggregator      │
│                                                └→ C ─┘                  │
│                                                                         │
│  3. ROUTER/DISPATCHER               4. ITERATIVE REFINEMENT             │
│     Input → Router ──┬→ A              Input → Generator ─┐             │
│                      ├→ B                        ↑        │             │
│                      └→ C                        └─ Critic←┘            │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
cd src/typescript

# Install dependencies
npm install

# Run individual demos
npm run demo:sequential
npm run demo:parallel
npm run demo:router
npm run demo:iterative

# Run all demos
npm run demo:all
```

---

## Pattern 1: Sequential Chain

**Output from one agent feeds into the next.**

```typescript
import { SequentialChain, Agent } from "./01_sequential_chain/chain.js";

const chain = new SequentialChain();
chain
  .addStep(researcher)
  .addStep(writer, (researchOutput) => `Write based on: ${researchOutput}`)
  .addStep(editor)
  .addStep(formatter);

const result = await chain.run("AI in Healthcare");
```

### Demo Results

| Agent | Tokens (in/out) | Time |
|-------|-----------------|------|
| Researcher | 60/500 | 4.2s |
| Writer | 564/653 | 5.4s |
| Editor | 722/663 | 5.2s |
| Formatter | 738/409 | 3.0s |
| **Total** | **2084/2225** | **17.8s** |

### When to Use
- Multi-step content creation (research → write → edit → format)
- Progressive refinement pipelines
- Each step builds on previous output

---

## Pattern 2: Parallel Fan-out

**Multiple agents process the same input concurrently.**

```typescript
import { ParallelFanOut } from "./02_parallel_fanout/fanout.js";

const fanout = new ParallelFanOut();
fanout
  .addParallelAgent(financialAnalyst)
  .addParallelAgent(technicalExpert)
  .addParallelAgent(marketResearcher)
  .addParallelAgent(riskAssessor)
  .setAggregator(strategicAdvisor);

const result = await fanout.run(businessDecision);
```

### Demo Results

| Metric | Sequential | Parallel |
|--------|------------|----------|
| 4 Expert Analyses | 14.82s | 4.89s |
| **Speedup** | 1x | **3.03x** |

### When to Use
- Multiple independent perspectives needed
- Time-critical decisions
- Diverse expert analysis

---

## Pattern 3: Router/Dispatcher

**Intelligent routing to specialized agents.**

```typescript
import { RouterDispatcher } from "./03_router_dispatcher/router.js";

const router = new RouterDispatcher();
router
  .setRouter(classifierAgent, ["technical", "billing", "sales", "general"])
  .addRoute("technical", technicalSupport)
  .addRoute("billing", billingSupport)
  .addRoute("sales", salesSupport)
  .addRoute("general", generalSupport)
  .setDefaultRoute("general");

const result = await router.run(customerMessage);
```

### Demo Results

| Test Case | Route | Confidence |
|-----------|-------|------------|
| API 500 errors | technical | 90% |
| Duplicate charge | billing | 90% |
| Enterprise pricing | sales | 90% |
| General feedback | general | 80% |

**Routing Accuracy: 100%**

### When to Use
- Customer support routing
- Task classification and delegation
- Specialized expert selection

---

## Pattern 4: Iterative Refinement

**Generator-critic loop for quality improvement.**

```typescript
import { IterativeRefinement } from "./04_iterative_refinement/refinement.js";

const refinement = new IterativeRefinement({
  maxIterations: 4,
  scoreThreshold: 0.85,
});

refinement
  .setGenerator(codeGenerator)
  .setCritic(codeReviewer);

const result = await refinement.run(codeRequest);
// result.iterations: [{score: 0.7}, {score: 0.85}]
// result.finalScore: 0.90
```

### Demo Results

| Iteration | Score | Status |
|-----------|-------|--------|
| 1 | 90% | Approved |

**First-pass quality was high, no refinement needed.**

### When to Use
- Code generation with review
- Content creation with editing
- Any task requiring quality gates

---

## Pattern Comparison

| Pattern | Latency | Parallelism | Use Case |
|---------|---------|-------------|----------|
| Sequential | Sum of all | None | Build-on-previous tasks |
| Parallel | Max of all | Full | Independent analyses |
| Router | Router + Handler | Handler only | Classification + handling |
| Iterative | Variable | None | Quality-critical output |

---

## Project Structure

```
src/typescript/
├── package.json
├── tsconfig.json
├── README.md
└── src/
    ├── shared/
    │   └── llm-client.ts      # LLM client, Agent class, metrics
    ├── 01_sequential_chain/
    │   ├── chain.ts           # SequentialChain orchestrator
    │   └── demo.ts            # Content creation demo
    ├── 02_parallel_fanout/
    │   ├── fanout.ts          # ParallelFanOut orchestrator
    │   └── demo.ts            # Business analysis demo
    ├── 03_router_dispatcher/
    │   ├── router.ts          # RouterDispatcher orchestrator
    │   └── demo.ts            # Customer support demo
    └── 04_iterative_refinement/
        ├── refinement.ts      # IterativeRefinement orchestrator
        └── demo.ts            # Code generation demo
```

---

## Key Concepts

### Agent

A specialized LLM instance with a specific role:

```typescript
const agent = new Agent(client, {
  name: "CodeReviewer",
  systemPrompt: "You are a senior code reviewer...",
  temperature: 0.2,  // Lower for consistency
  maxTokens: 500,
});
```

### Metrics Tracking

Built-in metrics for all patterns:

```typescript
result.metrics.printSummary("Pattern Name");
// Total Latency: 17.81s
// Total LLM Calls: 4
// Total Input Tokens: 2084
// Total Output Tokens: 2225
```

### Error Handling

Patterns handle failures gracefully:
- Router has fallback routes
- Parallel fan-out continues if one agent fails
- Iterative refinement has max iterations cap

---

## Best Practices

### 1. Temperature by Task

```typescript
// High temperature: Creative tasks
const writer = new Agent(client, { temperature: 0.7 });

// Low temperature: Precise tasks
const reviewer = new Agent(client, { temperature: 0.2 });
```

### 2. Token Limits

```typescript
// Balance for parallel agents
agent1: { maxTokens: 400 }
agent2: { maxTokens: 400 }
agent3: { maxTokens: 400 }
// All complete around same time
```

### 3. Context Passing

```typescript
// Use inputTransform for context
chain.addStep(editor, (previousOutput, originalInput) => {
  return `Original request: ${originalInput}\n\nDraft to edit: ${previousOutput}`;
});
```

---

## Production Considerations

### Rate Limiting

```typescript
// Parallel fan-out may hit rate limits
// Consider batching or delays
const delay = (ms: number) => new Promise(r => setTimeout(r, ms));
```

### Cost Optimization

```typescript
// Track cost per pattern
const cost = client.calculateCost(inputTokens, outputTokens);
// Choose pattern based on cost/quality tradeoff
```

### Caching

```typescript
// Cache common routing decisions
const routeCache = new Map<string, string>();
```

---

## Combining Patterns

Patterns can be nested:

```typescript
// Router → Sequential Chain
router.addRoute("complex", complexChain);

// Parallel → Iterative Refinement
// Each parallel agent uses refinement loop
```

---

## Summary

| Module | Pattern | Key Feature | Demo |
|--------|---------|-------------|------|
| 01 | Sequential Chain | Pipeline processing | Blog post creation |
| 02 | Parallel Fan-out | 3x speedup | Business analysis |
| 03 | Router/Dispatcher | 100% routing accuracy | Support tickets |
| 04 | Iterative Refinement | Quality gates | Code generation |

All patterns demonstrate practical multi-agent orchestration with Claude, complete with metrics tracking and error handling.
