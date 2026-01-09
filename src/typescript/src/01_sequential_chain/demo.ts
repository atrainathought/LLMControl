#!/usr/bin/env npx tsx
/**
 * Sequential Chain Demo
 *
 * Demonstrates a content creation pipeline where each agent
 * builds on the previous agent's work.
 */

import { LLMClient } from "../shared/llm-client.js";
import { createContentPipeline } from "./chain.js";

async function main() {
  console.log("=".repeat(70));
  console.log("SEQUENTIAL CHAIN PATTERN DEMO");
  console.log("=".repeat(70));
  console.log("\nPattern: Input → Researcher → Writer → Editor → Formatter → Output");
  console.log("\nTask: Create a blog post about 'The Future of AI in Healthcare'");

  const client = new LLMClient();
  const pipeline = createContentPipeline(client);

  console.log("\n" + "-".repeat(70));
  console.log("EXECUTING PIPELINE");
  console.log("-".repeat(70));

  const result = await pipeline.run("The Future of AI in Healthcare");

  // Show intermediate outputs
  console.log("\n" + "=".repeat(70));
  console.log("INTERMEDIATE OUTPUTS");
  console.log("=".repeat(70));

  for (const step of result.intermediateOutputs) {
    console.log(`\n--- ${step.agentName} ---`);
    console.log(`Output (first 300 chars):\n${step.output.slice(0, 300)}...`);
  }

  // Show final output
  console.log("\n" + "=".repeat(70));
  console.log("FINAL OUTPUT");
  console.log("=".repeat(70));
  console.log(result.finalOutput);

  // Show metrics
  result.metrics.printSummary("Sequential Chain");

  // Key insights
  console.log("\n" + "=".repeat(70));
  console.log("KEY INSIGHTS");
  console.log("=".repeat(70));
  console.log(`
SEQUENTIAL CHAIN PATTERN:

1. WHEN TO USE
   - Multi-step processing (research → write → edit)
   - Quality improvement through refinement
   - Complex tasks broken into specialized steps

2. ADVANTAGES
   - Clear, linear flow
   - Easy to debug (inspect each step)
   - Specialized agents for each task

3. DISADVANTAGES
   - Total latency = sum of all step latencies
   - Single point of failure (one bad step affects all)
   - Cannot parallelize independent work

4. BEST PRACTICES
   - Use inputTransform to provide context between steps
   - Lower temperature for editing/formatting (precision)
   - Higher temperature for creative steps (variety)
`);
}

main().catch(console.error);
