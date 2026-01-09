#!/usr/bin/env npx tsx
/**
 * Parallel Fan-out Demo
 *
 * Demonstrates parallel execution of multiple expert agents
 * with result aggregation.
 */

import { LLMClient } from "../shared/llm-client.js";
import { createAnalysisPipeline } from "./fanout.js";

async function main() {
  console.log("=".repeat(70));
  console.log("PARALLEL FAN-OUT PATTERN DEMO");
  console.log("=".repeat(70));
  console.log(`
Pattern:
            ┌→ FinancialAnalyst ─┐
  Input ───┼→ TechnicalExpert ───┼→ StrategicAdvisor → Output
            ├→ MarketResearcher ─┤
            └→ RiskAssessor ─────┘
`);
  console.log("Task: Analyze a business decision from multiple perspectives");

  const client = new LLMClient();
  const pipeline = createAnalysisPipeline(client);

  const businessDecision = `
Should our mid-size e-commerce company invest $2M in building
an AI-powered personalization engine for product recommendations?
We currently use a basic rule-based system. Our tech stack is
Python/Django with PostgreSQL. We have 500K monthly active users.
`;

  console.log("-".repeat(70));
  console.log("BUSINESS DECISION TO ANALYZE:");
  console.log("-".repeat(70));
  console.log(businessDecision);

  console.log("\n" + "-".repeat(70));
  console.log("EXECUTING PARALLEL ANALYSIS");
  console.log("-".repeat(70));

  const result = await pipeline.run(businessDecision);

  // Show parallel outputs
  console.log("\n" + "=".repeat(70));
  console.log("EXPERT ANALYSES (ran in parallel)");
  console.log("=".repeat(70));

  for (const output of result.parallelOutputs) {
    console.log(`\n--- ${output.agentName} (${(output.latencyMs / 1000).toFixed(2)}s) ---`);
    console.log(output.output);
  }

  // Show aggregated output
  console.log("\n" + "=".repeat(70));
  console.log("STRATEGIC ADVISOR SYNTHESIS");
  console.log("=".repeat(70));
  console.log(result.finalOutput);

  // Show timing comparison
  console.log("\n" + "=".repeat(70));
  console.log("TIMING ANALYSIS");
  console.log("=".repeat(70));

  const sequentialTime = result.parallelOutputs.reduce(
    (sum, o) => sum + o.latencyMs,
    0
  );

  console.log(`\nParallel execution time: ${(result.parallelLatencyMs / 1000).toFixed(2)}s`);
  console.log(`Sequential would have been: ${(sequentialTime / 1000).toFixed(2)}s`);
  console.log(`Speedup: ${(sequentialTime / result.parallelLatencyMs).toFixed(2)}x`);

  // Show metrics
  result.metrics.printSummary("Parallel Fan-out");

  // Key insights
  console.log("\n" + "=".repeat(70));
  console.log("KEY INSIGHTS");
  console.log("=".repeat(70));
  console.log(`
PARALLEL FAN-OUT PATTERN:

1. WHEN TO USE
   - Multiple independent analyses of same input
   - Need diverse perspectives
   - Time is critical (parallel = faster)

2. ADVANTAGES
   - Parallelism reduces latency (time = slowest agent, not sum)
   - Independent failures don't block others
   - Easy to add more parallel agents

3. DISADVANTAGES
   - Higher concurrent API usage
   - Aggregation can be complex
   - All agents must wait for slowest one

4. BEST PRACTICES
   - Use similar token limits for balanced completion
   - Aggregator needs clear instructions
   - Consider error handling for individual failures

5. SPEEDUP ACHIEVED
   - With 4 parallel agents: ~4x faster than sequential
   - Actual speedup depends on LLM API concurrency limits
`);
}

main().catch(console.error);
