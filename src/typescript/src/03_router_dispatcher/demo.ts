#!/usr/bin/env npx tsx
/**
 * Router/Dispatcher Demo
 *
 * Demonstrates intelligent routing of customer inquiries
 * to specialized support agents.
 */

import { LLMClient } from "../shared/llm-client.js";
import { createSupportRouter } from "./router.js";

async function main() {
  console.log("=".repeat(70));
  console.log("ROUTER/DISPATCHER PATTERN DEMO");
  console.log("=".repeat(70));
  console.log(`
Pattern:
                    ┌→ TechnicalSupport
  Input → Router ───┼→ BillingSupport
                    ├→ SalesSupport
                    └→ GeneralSupport
`);
  console.log("Task: Route customer inquiries to appropriate specialists\n");

  const client = new LLMClient();
  const router = createSupportRouter(client);

  // Test cases covering different routes
  const testCases = [
    {
      name: "Technical Issue",
      input: "The API is returning 500 errors when I try to authenticate. I'm using the Python SDK version 2.3.1 and my code worked fine yesterday. Here's the error: ConnectionResetError",
    },
    {
      name: "Billing Question",
      input: "I was charged twice this month for my subscription. Order ID #12345. I need a refund for the duplicate charge. My card ending in 4242 was charged $49.99 on both Jan 1 and Jan 3.",
    },
    {
      name: "Sales Inquiry",
      input: "We're a company of 500 employees looking to switch from Competitor X. What enterprise pricing options do you offer? We need SSO integration and dedicated support.",
    },
    {
      name: "General Feedback",
      input: "Just wanted to say I love the new dashboard design! The dark mode is great. Any chance you could add a mobile app in the future?",
    },
  ];

  const results: Array<{
    testName: string;
    route: string;
    confidence: number;
  }> = [];

  for (const testCase of testCases) {
    console.log("\n" + "=".repeat(70));
    console.log(`TEST: ${testCase.name}`);
    console.log("=".repeat(70));
    console.log(`\nCustomer: "${testCase.input.slice(0, 100)}..."\n`);

    const result = await router.run(testCase.input);

    results.push({
      testName: testCase.name,
      route: result.route,
      confidence: result.routingDecision.confidence,
    });

    console.log("\n" + "-".repeat(70));
    console.log("AGENT RESPONSE:");
    console.log("-".repeat(70));
    console.log(result.agentOutput);
  }

  // Summary
  console.log("\n" + "=".repeat(70));
  console.log("ROUTING SUMMARY");
  console.log("=".repeat(70));
  console.log("\n| Test Case         | Route           | Confidence |");
  console.log("|-------------------|-----------------|------------|");

  for (const r of results) {
    console.log(
      `| ${r.testName.padEnd(17)} | ${r.route.padEnd(15)} | ${(r.confidence * 100).toFixed(0)}%        |`
    );
  }

  // Show metrics
  router.getMetrics().printSummary("Router/Dispatcher");

  // Key insights
  console.log("\n" + "=".repeat(70));
  console.log("KEY INSIGHTS");
  console.log("=".repeat(70));
  console.log(`
ROUTER/DISPATCHER PATTERN:

1. WHEN TO USE
   - Multiple specialized handlers for different request types
   - Need intelligent classification before processing
   - Want to scale specialized expertise

2. ADVANTAGES
   - Requests go to the best-suited agent
   - Easy to add new routes/specialists
   - Router provides transparency (why this route?)

3. DISADVANTAGES
   - Router adds latency overhead
   - Misrouting leads to poor responses
   - Need good route definitions

4. BEST PRACTICES
   - Use low temperature for routing (consistency)
   - Include confidence scores
   - Always have a default/fallback route
   - Consider multi-label routing for complex queries

5. ROUTING ACCURACY
   - Clear route definitions = better classification
   - Include examples in router prompt if needed
   - Monitor misroutes and improve definitions
`);
}

main().catch(console.error);
