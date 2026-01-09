/**
 * Sequential Chain Pattern
 *
 * Agents are executed in sequence, where each agent's output
 * becomes the input for the next agent.
 *
 * Use Case: Multi-step content processing pipeline
 * - Research → Draft → Edit → Format
 *
 * Pattern:
 *   Input → Agent1 → Agent2 → Agent3 → Output
 */

import { Agent, LLMClient, OrchestrationMetrics } from "../shared/llm-client.js";

export interface ChainStep {
  agent: Agent;
  inputTransform?: (previousOutput: string, originalInput: string) => string;
}

export interface ChainResult {
  finalOutput: string;
  intermediateOutputs: Array<{
    agentName: string;
    input: string;
    output: string;
  }>;
  metrics: OrchestrationMetrics;
}

/**
 * Sequential Chain Orchestrator
 *
 * Executes agents in order, passing output from one to the next.
 */
export class SequentialChain {
  private steps: ChainStep[] = [];
  private metrics: OrchestrationMetrics;

  constructor() {
    this.metrics = new OrchestrationMetrics();
  }

  /**
   * Add an agent to the chain
   */
  addStep(
    agent: Agent,
    inputTransform?: (previousOutput: string, originalInput: string) => string
  ): SequentialChain {
    this.steps.push({ agent, inputTransform });
    this.metrics.addAgent(agent);
    return this;
  }

  /**
   * Execute the chain
   */
  async run(input: string): Promise<ChainResult> {
    this.metrics.start();

    const intermediateOutputs: ChainResult["intermediateOutputs"] = [];
    let currentInput = input;
    const originalInput = input;

    for (const step of this.steps) {
      // Transform input if transformer provided
      if (step.inputTransform) {
        currentInput = step.inputTransform(currentInput, originalInput);
      }

      console.log(`\n[${step.agent.name}] Processing...`);

      // Run the agent
      const output = await step.agent.run(currentInput);

      intermediateOutputs.push({
        agentName: step.agent.name,
        input: currentInput,
        output,
      });

      // Output becomes next input
      currentInput = output;
    }

    this.metrics.end();

    return {
      finalOutput: currentInput,
      intermediateOutputs,
      metrics: this.metrics,
    };
  }

  getMetrics(): OrchestrationMetrics {
    return this.metrics;
  }
}

/**
 * Create a content creation pipeline
 *
 * This example creates a blog post through multiple stages:
 * 1. Researcher - Gathers key points about the topic
 * 2. Writer - Creates a draft from the research
 * 3. Editor - Refines and improves the draft
 * 4. Formatter - Adds structure and formatting
 */
export function createContentPipeline(client: LLMClient): SequentialChain {
  const chain = new SequentialChain();

  // Step 1: Researcher
  const researcher = new Agent(client, {
    name: "Researcher",
    systemPrompt: `You are a research specialist. Given a topic, provide 5-7 key points
that should be covered in a blog post. Include interesting facts, statistics,
or examples where relevant. Be concise and factual.`,
    temperature: 0.5,
    maxTokens: 500,
  });

  // Step 2: Writer
  const writer = new Agent(client, {
    name: "Writer",
    systemPrompt: `You are a skilled blog writer. Given research notes, write an engaging
blog post draft. Include an introduction, body paragraphs covering each key point,
and a conclusion. Write in a conversational but professional tone.`,
    temperature: 0.7,
    maxTokens: 800,
  });

  // Step 3: Editor
  const editor = new Agent(client, {
    name: "Editor",
    systemPrompt: `You are a professional editor. Review the draft and improve it by:
1. Fixing any grammatical errors
2. Improving sentence flow and transitions
3. Strengthening weak arguments
4. Ensuring consistent tone
Return the improved version of the full post.`,
    temperature: 0.3,
    maxTokens: 800,
  });

  // Step 4: Formatter
  const formatter = new Agent(client, {
    name: "Formatter",
    systemPrompt: `You are a content formatter. Take the edited post and add:
1. A catchy title (# heading)
2. Section headings (## headings)
3. Bullet points where appropriate
4. A TL;DR summary at the end
Return the fully formatted markdown post.`,
    temperature: 0.2,
    maxTokens: 1000,
  });

  // Build the chain
  chain
    .addStep(researcher)
    .addStep(writer, (researchOutput) => {
      return `Based on this research, write a blog post:\n\n${researchOutput}`;
    })
    .addStep(editor, (draft) => {
      return `Please edit and improve this draft:\n\n${draft}`;
    })
    .addStep(formatter, (editedDraft) => {
      return `Please format this post with markdown:\n\n${editedDraft}`;
    });

  return chain;
}

/**
 * Create a code review pipeline
 *
 * 1. Analyzer - Identifies issues in code
 * 2. Suggester - Proposes improvements
 * 3. Implementer - Writes the fixed code
 */
export function createCodeReviewPipeline(client: LLMClient): SequentialChain {
  const chain = new SequentialChain();

  const analyzer = new Agent(client, {
    name: "CodeAnalyzer",
    systemPrompt: `You are a code analysis expert. Analyze the provided code and identify:
1. Bugs or potential issues
2. Performance concerns
3. Security vulnerabilities
4. Code style problems
Be specific and cite line numbers or specific code sections.`,
    temperature: 0.2,
    maxTokens: 600,
  });

  const suggester = new Agent(client, {
    name: "ImprovementSuggester",
    systemPrompt: `You are a senior developer. Given a code analysis report, propose
specific improvements for each issue identified. Prioritize by severity
and provide clear, actionable suggestions.`,
    temperature: 0.4,
    maxTokens: 600,
  });

  const implementer = new Agent(client, {
    name: "CodeImplementer",
    systemPrompt: `You are an expert programmer. Given improvement suggestions,
write the corrected code. Show the complete fixed version with comments
explaining each change.`,
    temperature: 0.2,
    maxTokens: 800,
  });

  chain
    .addStep(analyzer)
    .addStep(suggester, (analysis, originalCode) => {
      return `Original code:\n${originalCode}\n\nAnalysis:\n${analysis}\n\nSuggest improvements:`;
    })
    .addStep(implementer, (suggestions, originalCode) => {
      return `Original code:\n${originalCode}\n\nSuggestions:\n${suggestions}\n\nImplement fixes:`;
    });

  return chain;
}
