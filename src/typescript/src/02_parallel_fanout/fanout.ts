/**
 * Parallel Fan-out Pattern
 *
 * Multiple agents process the same input concurrently,
 * then results are aggregated by a final agent.
 *
 * Use Case: Multi-perspective analysis
 * - Get opinions from different expert viewpoints
 * - Parallel research from different angles
 *
 * Pattern:
 *            ┌→ Agent1 ─┐
 *   Input ──┼→ Agent2 ──┼→ Aggregator → Output
 *            └→ Agent3 ─┘
 */

import { Agent, LLMClient, OrchestrationMetrics } from "../shared/llm-client.js";

export interface FanOutResult {
  finalOutput: string;
  parallelOutputs: Array<{
    agentName: string;
    output: string;
    latencyMs: number;
  }>;
  parallelLatencyMs: number;
  totalLatencyMs: number;
  metrics: OrchestrationMetrics;
}

/**
 * Parallel Fan-out Orchestrator
 *
 * Runs multiple agents in parallel, then aggregates results.
 */
export class ParallelFanOut {
  private parallelAgents: Agent[] = [];
  private aggregator: Agent | null = null;
  private metrics: OrchestrationMetrics;

  constructor() {
    this.metrics = new OrchestrationMetrics();
  }

  /**
   * Add a parallel agent
   */
  addParallelAgent(agent: Agent): ParallelFanOut {
    this.parallelAgents.push(agent);
    this.metrics.addAgent(agent);
    return this;
  }

  /**
   * Set the aggregator agent
   */
  setAggregator(agent: Agent): ParallelFanOut {
    this.aggregator = agent;
    this.metrics.addAgent(agent);
    return this;
  }

  /**
   * Execute the fan-out pattern
   */
  async run(input: string): Promise<FanOutResult> {
    if (!this.aggregator) {
      throw new Error("Aggregator agent not set");
    }

    this.metrics.start();

    console.log(`\n[FanOut] Starting ${this.parallelAgents.length} parallel agents...`);

    // Run all agents in parallel
    const parallelStart = performance.now();

    const parallelPromises = this.parallelAgents.map(async (agent) => {
      const agentStart = performance.now();
      const output = await agent.run(input);
      const agentLatency = performance.now() - agentStart;

      console.log(`  [${agent.name}] Completed in ${(agentLatency / 1000).toFixed(2)}s`);

      return {
        agentName: agent.name,
        output,
        latencyMs: agentLatency,
      };
    });

    const parallelOutputs = await Promise.all(parallelPromises);
    const parallelLatencyMs = performance.now() - parallelStart;

    console.log(`\n[FanOut] All parallel agents completed in ${(parallelLatencyMs / 1000).toFixed(2)}s`);

    // Aggregate results
    console.log(`\n[${this.aggregator.name}] Aggregating results...`);

    const aggregationInput = this.formatForAggregation(input, parallelOutputs);
    const finalOutput = await this.aggregator.run(aggregationInput);

    this.metrics.end();

    return {
      finalOutput,
      parallelOutputs,
      parallelLatencyMs,
      totalLatencyMs: performance.now() - parallelStart,
      metrics: this.metrics,
    };
  }

  private formatForAggregation(
    originalInput: string,
    outputs: Array<{ agentName: string; output: string }>
  ): string {
    let formatted = `Original request: ${originalInput}\n\n`;
    formatted += "Expert analyses:\n\n";

    for (const { agentName, output } of outputs) {
      formatted += `--- ${agentName} ---\n${output}\n\n`;
    }

    return formatted;
  }

  getMetrics(): OrchestrationMetrics {
    return this.metrics;
  }
}

/**
 * Create a multi-perspective analysis pipeline
 *
 * Example: Analyze a business decision from multiple angles
 * - Financial analyst
 * - Technical expert
 * - Market researcher
 * - Risk assessor
 */
export function createAnalysisPipeline(client: LLMClient): ParallelFanOut {
  const fanout = new ParallelFanOut();

  // Parallel experts
  const financialAnalyst = new Agent(client, {
    name: "FinancialAnalyst",
    systemPrompt: `You are a financial analyst. Analyze the given business decision
from a financial perspective. Consider:
- ROI and payback period
- Cash flow implications
- Budget requirements
- Financial risks
Provide a concise financial assessment.`,
    temperature: 0.3,
    maxTokens: 400,
  });

  const technicalExpert = new Agent(client, {
    name: "TechnicalExpert",
    systemPrompt: `You are a technical expert. Analyze the given business decision
from a technical perspective. Consider:
- Technical feasibility
- Infrastructure requirements
- Integration challenges
- Technical risks
Provide a concise technical assessment.`,
    temperature: 0.3,
    maxTokens: 400,
  });

  const marketResearcher = new Agent(client, {
    name: "MarketResearcher",
    systemPrompt: `You are a market researcher. Analyze the given business decision
from a market perspective. Consider:
- Market opportunity
- Competitive landscape
- Customer demand
- Timing considerations
Provide a concise market assessment.`,
    temperature: 0.3,
    maxTokens: 400,
  });

  const riskAssessor = new Agent(client, {
    name: "RiskAssessor",
    systemPrompt: `You are a risk management specialist. Analyze the given business decision
from a risk perspective. Consider:
- Operational risks
- Strategic risks
- Regulatory/compliance risks
- Mitigation strategies
Provide a concise risk assessment.`,
    temperature: 0.3,
    maxTokens: 400,
  });

  // Aggregator
  const strategicAdvisor = new Agent(client, {
    name: "StrategicAdvisor",
    systemPrompt: `You are a senior strategic advisor. Given multiple expert analyses,
synthesize them into a comprehensive recommendation. Include:
1. Summary of key insights from each perspective
2. Areas of agreement and conflict
3. Overall recommendation (proceed/delay/reject)
4. Key conditions or next steps
Be balanced and consider all viewpoints.`,
    temperature: 0.4,
    maxTokens: 600,
  });

  fanout
    .addParallelAgent(financialAnalyst)
    .addParallelAgent(technicalExpert)
    .addParallelAgent(marketResearcher)
    .addParallelAgent(riskAssessor)
    .setAggregator(strategicAdvisor);

  return fanout;
}

/**
 * Create a multi-language translation pipeline
 *
 * Translate content to multiple languages in parallel
 */
export function createTranslationPipeline(client: LLMClient): ParallelFanOut {
  const fanout = new ParallelFanOut();

  const languages = ["Spanish", "French", "German", "Japanese"];

  for (const language of languages) {
    const translator = new Agent(client, {
      name: `${language}Translator`,
      systemPrompt: `You are a professional translator specializing in ${language}.
Translate the given text to ${language}. Maintain the original tone and meaning.
Only output the translation, no explanations.`,
      temperature: 0.2,
      maxTokens: 500,
    });
    fanout.addParallelAgent(translator);
  }

  const aggregator = new Agent(client, {
    name: "TranslationAggregator",
    systemPrompt: `You are a localization coordinator. Given translations in multiple languages,
create a summary showing:
1. The original text
2. Each translation with its language label
Format nicely for easy review.`,
    temperature: 0.1,
    maxTokens: 800,
  });

  fanout.setAggregator(aggregator);

  return fanout;
}
