/**
 * Iterative Refinement Pattern
 *
 * A generator agent creates output, then a critic agent
 * evaluates it and provides feedback. This loops until
 * the output meets quality criteria.
 *
 * Use Case: Quality-focused generation
 * - Code generation with testing
 * - Content creation with review
 * - Design with validation
 *
 * Pattern:
 *   Input → Generator ─┬─→ Output
 *              ↑       │
 *              └─ Critic ←─┘
 *           (repeat until satisfied)
 */

import { Agent, LLMClient, OrchestrationMetrics } from "../shared/llm-client.js";

export interface CriticFeedback {
  approved: boolean;
  score: number;
  issues: string[];
  suggestions: string[];
}

export interface RefinementIteration {
  iteration: number;
  generatorOutput: string;
  criticFeedback: CriticFeedback;
}

export interface RefinementResult {
  finalOutput: string;
  iterations: RefinementIteration[];
  totalIterations: number;
  finalScore: number;
  metrics: OrchestrationMetrics;
}

/**
 * Iterative Refinement Orchestrator
 *
 * Loops between generator and critic until quality threshold is met.
 */
export class IterativeRefinement {
  private generator: Agent | null = null;
  private critic: Agent | null = null;
  private maxIterations: number = 5;
  private scoreThreshold: number = 0.8;
  private metrics: OrchestrationMetrics;

  constructor(options?: { maxIterations?: number; scoreThreshold?: number }) {
    this.maxIterations = options?.maxIterations ?? 5;
    this.scoreThreshold = options?.scoreThreshold ?? 0.8;
    this.metrics = new OrchestrationMetrics();
  }

  /**
   * Set the generator agent
   */
  setGenerator(agent: Agent): IterativeRefinement {
    this.generator = agent;
    this.metrics.addAgent(agent);
    return this;
  }

  /**
   * Set the critic agent
   */
  setCritic(agent: Agent): IterativeRefinement {
    this.critic = agent;
    this.metrics.addAgent(agent);
    return this;
  }

  /**
   * Execute the refinement loop
   */
  async run(input: string): Promise<RefinementResult> {
    if (!this.generator || !this.critic) {
      throw new Error("Generator and critic agents must be set");
    }

    this.metrics.start();

    const iterations: RefinementIteration[] = [];
    let currentOutput = "";
    let previousFeedback: CriticFeedback | null = null;
    let finalScore = 0;

    for (let i = 1; i <= this.maxIterations; i++) {
      console.log(`\n--- Iteration ${i}/${this.maxIterations} ---`);

      // Generate (with feedback from previous iteration if available)
      console.log(`[${this.generator.name}] Generating...`);

      let generatorPrompt = input;
      if (previousFeedback) {
        generatorPrompt = this.formatGeneratorPrompt(
          input,
          currentOutput,
          previousFeedback
        );
      }

      currentOutput = await this.generator.run(generatorPrompt);
      console.log(`[${this.generator.name}] Generated ${currentOutput.length} chars`);

      // Critique
      console.log(`[${this.critic.name}] Evaluating...`);

      const criticPrompt = this.formatCriticPrompt(input, currentOutput);
      const criticResponse = await this.critic.run(criticPrompt);
      const feedback = this.parseCriticFeedback(criticResponse);

      console.log(`[${this.critic.name}] Score: ${(feedback.score * 100).toFixed(0)}%`);
      console.log(`[${this.critic.name}] Issues: ${feedback.issues.length}`);

      iterations.push({
        iteration: i,
        generatorOutput: currentOutput,
        criticFeedback: feedback,
      });

      finalScore = feedback.score;
      previousFeedback = feedback;

      // Check if we've reached the threshold
      if (feedback.approved || feedback.score >= this.scoreThreshold) {
        console.log(`\n✓ Quality threshold met at iteration ${i}`);
        break;
      }

      if (i < this.maxIterations) {
        console.log(`  Continuing refinement...`);
      }
    }

    this.metrics.end();

    return {
      finalOutput: currentOutput,
      iterations,
      totalIterations: iterations.length,
      finalScore,
      metrics: this.metrics,
    };
  }

  private formatGeneratorPrompt(
    originalInput: string,
    previousOutput: string,
    feedback: CriticFeedback
  ): string {
    return `Original request: ${originalInput}

Your previous attempt:
${previousOutput}

Critic feedback:
Score: ${(feedback.score * 100).toFixed(0)}%
Issues: ${feedback.issues.join("; ")}
Suggestions: ${feedback.suggestions.join("; ")}

Please improve your response addressing the feedback above.`;
  }

  private formatCriticPrompt(originalInput: string, output: string): string {
    return `Original request: ${originalInput}

Generated output:
${output}

Evaluate this output and respond with a JSON object:
{
  "approved": true/false,
  "score": 0.0-1.0,
  "issues": ["issue 1", "issue 2"],
  "suggestions": ["suggestion 1", "suggestion 2"]
}`;
  }

  private parseCriticFeedback(response: string): CriticFeedback {
    try {
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          approved: parsed.approved ?? false,
          score: parsed.score ?? 0.5,
          issues: parsed.issues ?? [],
          suggestions: parsed.suggestions ?? [],
        };
      }
    } catch (e) {
      // Fallback parsing
    }

    // Fallback
    return {
      approved: false,
      score: 0.5,
      issues: ["Could not parse feedback"],
      suggestions: ["Try again with clearer structure"],
    };
  }

  getMetrics(): OrchestrationMetrics {
    return this.metrics;
  }
}

/**
 * Create a code generation refinement loop
 *
 * Generator writes code, critic reviews for bugs and style
 */
export function createCodeRefinement(client: LLMClient): IterativeRefinement {
  const refinement = new IterativeRefinement({
    maxIterations: 4,
    scoreThreshold: 0.85,
  });

  const codeGenerator = new Agent(client, {
    name: "CodeGenerator",
    systemPrompt: `You are an expert programmer. Generate clean, efficient code
that follows best practices. Include:
- Clear variable names
- Proper error handling
- Comments for complex logic
- Type annotations where applicable

When given feedback, carefully address each issue.`,
    temperature: 0.4,
    maxTokens: 800,
  });

  const codeReviewer = new Agent(client, {
    name: "CodeReviewer",
    systemPrompt: `You are a senior code reviewer. Evaluate code for:
- Correctness: Does it do what was requested?
- Style: Clean, readable, follows conventions?
- Efficiency: No obvious performance issues?
- Safety: Proper error handling?
- Completeness: All edge cases covered?

Be constructive but thorough. Score from 0 to 1.`,
    temperature: 0.2,
    maxTokens: 500,
  });

  refinement.setGenerator(codeGenerator).setCritic(codeReviewer);

  return refinement;
}

/**
 * Create a content writing refinement loop
 *
 * Generator writes content, critic reviews for quality
 */
export function createContentRefinement(client: LLMClient): IterativeRefinement {
  const refinement = new IterativeRefinement({
    maxIterations: 3,
    scoreThreshold: 0.8,
  });

  const contentWriter = new Agent(client, {
    name: "ContentWriter",
    systemPrompt: `You are a professional content writer. Create engaging,
well-structured content that:
- Has a clear introduction and conclusion
- Uses appropriate tone for the audience
- Includes relevant examples
- Flows logically between paragraphs

When given feedback, revise to address all issues.`,
    temperature: 0.7,
    maxTokens: 1000,
  });

  const contentEditor = new Agent(client, {
    name: "ContentEditor",
    systemPrompt: `You are a professional editor. Evaluate content for:
- Clarity: Is the message clear?
- Structure: Well-organized with good flow?
- Engagement: Does it hold attention?
- Grammar: Any errors or awkward phrasing?
- Completeness: Does it fully address the topic?

Be specific about issues and how to fix them.`,
    temperature: 0.3,
    maxTokens: 500,
  });

  refinement.setGenerator(contentWriter).setCritic(contentEditor);

  return refinement;
}

/**
 * Create a plan refinement loop
 *
 * Generator creates plans, critic evaluates feasibility
 */
export function createPlanRefinement(client: LLMClient): IterativeRefinement {
  const refinement = new IterativeRefinement({
    maxIterations: 3,
    scoreThreshold: 0.85,
  });

  const planGenerator = new Agent(client, {
    name: "PlanGenerator",
    systemPrompt: `You are a strategic planner. Create detailed, actionable plans that:
- Have clear milestones and deliverables
- Consider dependencies between tasks
- Include realistic time estimates
- Account for potential risks
- Specify required resources

When given feedback, refine the plan accordingly.`,
    temperature: 0.5,
    maxTokens: 800,
  });

  const planReviewer = new Agent(client, {
    name: "PlanReviewer",
    systemPrompt: `You are a project management expert. Evaluate plans for:
- Feasibility: Are the goals achievable?
- Completeness: All necessary steps included?
- Dependencies: Are they properly sequenced?
- Resources: Realistic requirements?
- Risks: Are they identified and mitigated?

Score the plan's quality from 0 to 1.`,
    temperature: 0.3,
    maxTokens: 500,
  });

  refinement.setGenerator(planGenerator).setCritic(planReviewer);

  return refinement;
}
