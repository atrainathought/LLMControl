#!/usr/bin/env npx tsx
/**
 * Iterative Refinement Demo
 *
 * Demonstrates the generator-critic loop for quality improvement.
 */

import { LLMClient } from "../shared/llm-client.js";
import { createCodeRefinement } from "./refinement.js";

async function main() {
  console.log("=".repeat(70));
  console.log("ITERATIVE REFINEMENT PATTERN DEMO");
  console.log("=".repeat(70));
  console.log(`
Pattern:
  Input → Generator ─┬─→ Output
             ↑       │
             └─ Critic ←─┘
          (repeat until satisfied)
`);
  console.log("Task: Generate and refine code through critic feedback\n");

  const client = new LLMClient();
  const refinement = createCodeRefinement(client);

  const codeRequest = `
Write a TypeScript function that:
1. Takes an array of numbers
2. Returns the top N largest unique values
3. Handles edge cases (empty array, N > unique count)
4. Is well-documented with examples
`;

  console.log("-".repeat(70));
  console.log("CODE REQUEST:");
  console.log("-".repeat(70));
  console.log(codeRequest);

  console.log("\n" + "-".repeat(70));
  console.log("REFINEMENT PROCESS");
  console.log("-".repeat(70));

  const result = await refinement.run(codeRequest);

  // Show iteration history
  console.log("\n" + "=".repeat(70));
  console.log("ITERATION HISTORY");
  console.log("=".repeat(70));

  for (const iter of result.iterations) {
    console.log(`\n--- Iteration ${iter.iteration} ---`);
    console.log(`Score: ${(iter.criticFeedback.score * 100).toFixed(0)}%`);
    console.log(`Approved: ${iter.criticFeedback.approved}`);

    if (iter.criticFeedback.issues.length > 0) {
      console.log(`Issues:`);
      iter.criticFeedback.issues.forEach((issue, i) => {
        console.log(`  ${i + 1}. ${issue}`);
      });
    }

    if (iter.criticFeedback.suggestions.length > 0) {
      console.log(`Suggestions:`);
      iter.criticFeedback.suggestions.forEach((sugg, i) => {
        console.log(`  ${i + 1}. ${sugg}`);
      });
    }
  }

  // Show final output
  console.log("\n" + "=".repeat(70));
  console.log("FINAL OUTPUT");
  console.log("=".repeat(70));
  console.log(result.finalOutput);

  // Quality progression
  console.log("\n" + "=".repeat(70));
  console.log("QUALITY PROGRESSION");
  console.log("=".repeat(70));
  console.log("\nIteration | Score");
  console.log("----------|------");

  for (const iter of result.iterations) {
    const bar = "█".repeat(Math.floor(iter.criticFeedback.score * 20));
    console.log(
      `    ${iter.iteration}     | ${(iter.criticFeedback.score * 100).toFixed(0).padStart(3)}% ${bar}`
    );
  }

  // Show metrics
  result.metrics.printSummary("Iterative Refinement");

  // Additional stats
  console.log(`\nRefinement Stats:`);
  console.log(`  Iterations: ${result.totalIterations}`);
  console.log(`  Final Score: ${(result.finalScore * 100).toFixed(0)}%`);
  console.log(`  LLM Calls: ${result.totalIterations * 2} (generator + critic each)`);

  // Key insights
  console.log("\n" + "=".repeat(70));
  console.log("KEY INSIGHTS");
  console.log("=".repeat(70));
  console.log(`
ITERATIVE REFINEMENT PATTERN:

1. WHEN TO USE
   - Quality is critical (code, legal docs, etc.)
   - First-pass output often needs improvement
   - Clear evaluation criteria exist

2. ADVANTAGES
   - Progressive quality improvement
   - Built-in quality measurement
   - Generator learns from feedback

3. DISADVANTAGES
   - Multiple iterations = higher cost
   - May oscillate without converging
   - Requires good critic prompts

4. BEST PRACTICES
   - Set maximum iterations to prevent infinite loops
   - Use score threshold to stop early
   - Critic should give actionable feedback
   - Lower temperature for critic (consistency)

5. CONVERGENCE
   - Good prompt design → fewer iterations
   - Track score progression to detect issues
   - Consider early stopping if score decreases
`);
}

main().catch(console.error);
