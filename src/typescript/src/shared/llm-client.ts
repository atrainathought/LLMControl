/**
 * Shared LLM Client for TypeScript Orchestration Patterns
 *
 * Provides a unified interface for calling Claude with:
 * - Token tracking
 * - Latency measurement
 * - Cost calculation
 */

import Anthropic from "@anthropic-ai/sdk";
import { config } from "dotenv";
import { resolve } from "path";

// Load environment variables from config/.env
config({ path: resolve(process.cwd(), "../../config/.env") });

// Also try the project root
config({ path: resolve(process.cwd(), "../../../config/.env") });

export interface LLMResponse {
  content: string;
  inputTokens: number;
  outputTokens: number;
  latencyMs: number;
  model: string;
}

export interface AgentConfig {
  name: string;
  systemPrompt: string;
  temperature?: number;
  maxTokens?: number;
}

export class LLMClient {
  private client: Anthropic;
  private model: string = "claude-3-haiku-20240307";

  // Pricing per 1M tokens (as of 2024)
  private pricing = {
    "claude-3-haiku-20240307": { input: 0.25, output: 1.25 },
    "claude-3-sonnet-20240229": { input: 3.0, output: 15.0 },
    "claude-3-opus-20240229": { input: 15.0, output: 75.0 },
  };

  constructor(model?: string) {
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      throw new Error(
        "ANTHROPIC_API_KEY not found. Set it in config/.env or environment."
      );
    }

    this.client = new Anthropic({ apiKey });
    if (model) {
      this.model = model;
    }
  }

  async complete(
    prompt: string,
    options: {
      systemPrompt?: string;
      temperature?: number;
      maxTokens?: number;
    } = {}
  ): Promise<LLMResponse> {
    const startTime = performance.now();

    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: options.maxTokens ?? 1024,
      temperature: options.temperature ?? 0.7,
      system: options.systemPrompt,
      messages: [{ role: "user", content: prompt }],
    });

    const latencyMs = performance.now() - startTime;

    const content =
      response.content[0].type === "text" ? response.content[0].text : "";

    return {
      content,
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
      latencyMs,
      model: this.model,
    };
  }

  async chat(
    messages: Array<{ role: "user" | "assistant"; content: string }>,
    options: {
      systemPrompt?: string;
      temperature?: number;
      maxTokens?: number;
    } = {}
  ): Promise<LLMResponse> {
    const startTime = performance.now();

    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: options.maxTokens ?? 1024,
      temperature: options.temperature ?? 0.7,
      system: options.systemPrompt,
      messages,
    });

    const latencyMs = performance.now() - startTime;

    const content =
      response.content[0].type === "text" ? response.content[0].text : "";

    return {
      content,
      inputTokens: response.usage.input_tokens,
      outputTokens: response.usage.output_tokens,
      latencyMs,
      model: this.model,
    };
  }

  calculateCost(inputTokens: number, outputTokens: number): number {
    const modelPricing =
      this.pricing[this.model as keyof typeof this.pricing] ||
      this.pricing["claude-3-haiku-20240307"];

    return (
      (inputTokens / 1_000_000) * modelPricing.input +
      (outputTokens / 1_000_000) * modelPricing.output
    );
  }

  getModel(): string {
    return this.model;
  }
}

/**
 * Agent class for multi-agent orchestration
 */
export class Agent {
  private client: LLMClient;
  private config: AgentConfig;
  private history: Array<{ role: "user" | "assistant"; content: string }> = [];

  // Track cumulative stats
  public totalInputTokens = 0;
  public totalOutputTokens = 0;
  public totalLatencyMs = 0;
  public callCount = 0;

  constructor(client: LLMClient, config: AgentConfig) {
    this.client = client;
    this.config = config;
  }

  get name(): string {
    return this.config.name;
  }

  async run(input: string): Promise<string> {
    const response = await this.client.complete(input, {
      systemPrompt: this.config.systemPrompt,
      temperature: this.config.temperature ?? 0.7,
      maxTokens: this.config.maxTokens ?? 1024,
    });

    // Track stats
    this.totalInputTokens += response.inputTokens;
    this.totalOutputTokens += response.outputTokens;
    this.totalLatencyMs += response.latencyMs;
    this.callCount++;

    return response.content;
  }

  async chat(input: string): Promise<string> {
    // Add user message to history
    this.history.push({ role: "user", content: input });

    const response = await this.client.chat(this.history, {
      systemPrompt: this.config.systemPrompt,
      temperature: this.config.temperature ?? 0.7,
      maxTokens: this.config.maxTokens ?? 1024,
    });

    // Add assistant response to history
    this.history.push({ role: "assistant", content: response.content });

    // Track stats
    this.totalInputTokens += response.inputTokens;
    this.totalOutputTokens += response.outputTokens;
    this.totalLatencyMs += response.latencyMs;
    this.callCount++;

    return response.content;
  }

  clearHistory(): void {
    this.history = [];
  }

  getStats(): {
    name: string;
    calls: number;
    inputTokens: number;
    outputTokens: number;
    latencyMs: number;
  } {
    return {
      name: this.config.name,
      calls: this.callCount,
      inputTokens: this.totalInputTokens,
      outputTokens: this.totalOutputTokens,
      latencyMs: this.totalLatencyMs,
    };
  }
}

/**
 * Metrics tracker for orchestration patterns
 */
export class OrchestrationMetrics {
  private agents: Agent[] = [];
  private startTime: number = 0;
  private endTime: number = 0;

  addAgent(agent: Agent): void {
    this.agents.push(agent);
  }

  start(): void {
    this.startTime = performance.now();
  }

  end(): void {
    this.endTime = performance.now();
  }

  getSummary(): {
    totalLatencyMs: number;
    totalInputTokens: number;
    totalOutputTokens: number;
    totalCalls: number;
    agentStats: ReturnType<Agent["getStats"]>[];
  } {
    const agentStats = this.agents.map((a) => a.getStats());

    return {
      totalLatencyMs: this.endTime - this.startTime,
      totalInputTokens: agentStats.reduce((sum, a) => sum + a.inputTokens, 0),
      totalOutputTokens: agentStats.reduce((sum, a) => sum + a.outputTokens, 0),
      totalCalls: agentStats.reduce((sum, a) => sum + a.calls, 0),
      agentStats,
    };
  }

  printSummary(patternName: string): void {
    const summary = this.getSummary();

    console.log("\n" + "=".repeat(70));
    console.log(`ORCHESTRATION METRICS: ${patternName}`);
    console.log("=".repeat(70));
    console.log(`\nTotal Latency: ${(summary.totalLatencyMs / 1000).toFixed(2)}s`);
    console.log(`Total LLM Calls: ${summary.totalCalls}`);
    console.log(`Total Input Tokens: ${summary.totalInputTokens}`);
    console.log(`Total Output Tokens: ${summary.totalOutputTokens}`);

    console.log("\nPer-Agent Breakdown:");
    console.log("-".repeat(60));
    for (const agent of summary.agentStats) {
      console.log(
        `  ${agent.name}: ${agent.calls} calls, ` +
          `${agent.inputTokens}/${agent.outputTokens} tokens, ` +
          `${(agent.latencyMs / 1000).toFixed(2)}s`
      );
    }
  }
}
