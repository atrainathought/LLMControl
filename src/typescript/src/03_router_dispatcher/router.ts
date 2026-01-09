/**
 * Router/Dispatcher Pattern
 *
 * A router agent analyzes the input and dispatches to
 * the most appropriate specialized agent.
 *
 * Use Case: Customer support routing, task delegation
 * - Route questions to appropriate experts
 * - Classify and handle different request types
 *
 * Pattern:
 *                    ┌→ TechnicalAgent
 *   Input → Router ──┼→ BillingAgent
 *                    └→ GeneralAgent
 */

import { Agent, LLMClient, OrchestrationMetrics } from "../shared/llm-client.js";

export interface RoutingResult {
  route: string;
  confidence: number;
  reasoning: string;
}

export interface RouterResult {
  route: string;
  routingDecision: RoutingResult;
  agentOutput: string;
  totalLatencyMs: number;
  metrics: OrchestrationMetrics;
}

/**
 * Router/Dispatcher Orchestrator
 *
 * Routes input to specialized agents based on classification.
 */
export class RouterDispatcher {
  private router: Agent | null = null;
  private routes: Map<string, Agent> = new Map();
  private defaultRoute: string | null = null;
  private metrics: OrchestrationMetrics;

  constructor() {
    this.metrics = new OrchestrationMetrics();
  }

  /**
   * Set the router agent
   */
  setRouter(agent: Agent, routeNames: string[]): RouterDispatcher {
    this.router = agent;
    this.metrics.addAgent(agent);

    // Update router's system prompt with available routes
    const routeList = routeNames.join(", ");
    (agent as any).config.systemPrompt += `

Available routes: ${routeList}
You must respond with ONLY a JSON object in this exact format:
{"route": "ROUTE_NAME", "confidence": 0.0-1.0, "reasoning": "brief explanation"}`;

    return this;
  }

  /**
   * Add a route with its handler agent
   */
  addRoute(routeName: string, agent: Agent): RouterDispatcher {
    this.routes.set(routeName, agent);
    this.metrics.addAgent(agent);
    return this;
  }

  /**
   * Set the default route for unmatched inputs
   */
  setDefaultRoute(routeName: string): RouterDispatcher {
    this.defaultRoute = routeName;
    return this;
  }

  /**
   * Execute the routing pattern
   */
  async run(input: string): Promise<RouterResult> {
    if (!this.router) {
      throw new Error("Router agent not set");
    }

    this.metrics.start();

    console.log("\n[Router] Analyzing input...");

    // Get routing decision
    const routingResponse = await this.router.run(input);
    const routingDecision = this.parseRoutingDecision(routingResponse);

    console.log(`[Router] Decision: ${routingDecision.route} (confidence: ${(routingDecision.confidence * 100).toFixed(0)}%)`);
    console.log(`[Router] Reasoning: ${routingDecision.reasoning}`);

    // Find the appropriate agent
    let targetAgent = this.routes.get(routingDecision.route);

    if (!targetAgent && this.defaultRoute) {
      console.log(`[Router] Route not found, using default: ${this.defaultRoute}`);
      targetAgent = this.routes.get(this.defaultRoute);
    }

    if (!targetAgent) {
      throw new Error(`No agent found for route: ${routingDecision.route}`);
    }

    // Execute the target agent
    console.log(`\n[${targetAgent.name}] Handling request...`);
    const agentOutput = await targetAgent.run(input);

    this.metrics.end();

    return {
      route: routingDecision.route,
      routingDecision,
      agentOutput,
      totalLatencyMs: this.metrics.getSummary().totalLatencyMs,
      metrics: this.metrics,
    };
  }

  private parseRoutingDecision(response: string): RoutingResult {
    try {
      // Try to extract JSON from the response
      const jsonMatch = response.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          route: parsed.route || "general",
          confidence: parsed.confidence || 0.5,
          reasoning: parsed.reasoning || "No reasoning provided",
        };
      }
    } catch (e) {
      // Fallback parsing
    }

    // Fallback: look for route keywords
    const routeNames = Array.from(this.routes.keys());
    for (const routeName of routeNames) {
      if (response.toLowerCase().includes(routeName.toLowerCase())) {
        return {
          route: routeName,
          confidence: 0.6,
          reasoning: "Extracted from text response",
        };
      }
    }

    return {
      route: this.defaultRoute || "general",
      confidence: 0.3,
      reasoning: "Could not parse routing decision",
    };
  }

  getMetrics(): OrchestrationMetrics {
    return this.metrics;
  }
}

/**
 * Create a customer support router
 *
 * Routes customer inquiries to appropriate specialists
 */
export function createSupportRouter(client: LLMClient): RouterDispatcher {
  const router = new RouterDispatcher();

  // Router agent
  const routerAgent = new Agent(client, {
    name: "SupportRouter",
    systemPrompt: `You are a customer support router. Analyze the customer's message
and determine which department should handle it:

- technical: Software bugs, feature issues, integration problems, API questions
- billing: Payment issues, subscription changes, invoices, refunds
- sales: Pricing questions, enterprise inquiries, feature comparisons
- general: General inquiries, feedback, other

Consider the urgency and specificity of the request.`,
    temperature: 0.1,
    maxTokens: 200,
  });

  // Specialized agents
  const technicalAgent = new Agent(client, {
    name: "TechnicalSupport",
    systemPrompt: `You are a technical support specialist. Help customers with:
- Troubleshooting software issues
- Explaining technical features
- Providing code examples
- Debugging integration problems
Be technical but clear. Ask clarifying questions if needed.`,
    temperature: 0.3,
    maxTokens: 500,
  });

  const billingAgent = new Agent(client, {
    name: "BillingSupport",
    systemPrompt: `You are a billing specialist. Help customers with:
- Understanding charges
- Processing refunds
- Managing subscriptions
- Invoice inquiries
Be precise with numbers and policies. Show empathy for billing issues.`,
    temperature: 0.2,
    maxTokens: 400,
  });

  const salesAgent = new Agent(client, {
    name: "SalesSupport",
    systemPrompt: `You are a sales specialist. Help customers with:
- Understanding pricing tiers
- Comparing features
- Enterprise solutions
- Custom packages
Be informative and helpful without being pushy.`,
    temperature: 0.4,
    maxTokens: 400,
  });

  const generalAgent = new Agent(client, {
    name: "GeneralSupport",
    systemPrompt: `You are a general support representative. Help customers with:
- General product information
- Company policies
- Feedback collection
- Any inquiries not fitting other categories
Be friendly and helpful.`,
    temperature: 0.5,
    maxTokens: 400,
  });

  router
    .setRouter(routerAgent, ["technical", "billing", "sales", "general"])
    .addRoute("technical", technicalAgent)
    .addRoute("billing", billingAgent)
    .addRoute("sales", salesAgent)
    .addRoute("general", generalAgent)
    .setDefaultRoute("general");

  return router;
}

/**
 * Create a task type router
 *
 * Routes different types of work tasks to specialized handlers
 */
export function createTaskRouter(client: LLMClient): RouterDispatcher {
  const router = new RouterDispatcher();

  const routerAgent = new Agent(client, {
    name: "TaskRouter",
    systemPrompt: `You are a task routing system. Analyze the request and determine
the appropriate handler:

- code: Programming, debugging, code review
- writing: Content creation, editing, summarization
- analysis: Data analysis, research, comparisons
- planning: Project planning, scheduling, organization

Choose based on the primary nature of the task.`,
    temperature: 0.1,
    maxTokens: 200,
  });

  const codeAgent = new Agent(client, {
    name: "CodeAssistant",
    systemPrompt: `You are an expert programmer. Handle coding tasks including:
- Writing new code
- Debugging issues
- Code review
- Architecture suggestions
Use markdown code blocks and be precise.`,
    temperature: 0.2,
    maxTokens: 800,
  });

  const writingAgent = new Agent(client, {
    name: "WritingAssistant",
    systemPrompt: `You are a professional writer. Handle writing tasks including:
- Creating content
- Editing and proofreading
- Summarization
- Style improvements
Match the requested tone and style.`,
    temperature: 0.6,
    maxTokens: 800,
  });

  const analysisAgent = new Agent(client, {
    name: "AnalysisAssistant",
    systemPrompt: `You are a research analyst. Handle analysis tasks including:
- Data interpretation
- Comparative analysis
- Research synthesis
- Insight generation
Be thorough and cite sources when available.`,
    temperature: 0.3,
    maxTokens: 800,
  });

  const planningAgent = new Agent(client, {
    name: "PlanningAssistant",
    systemPrompt: `You are a project planner. Handle planning tasks including:
- Project breakdown
- Timeline creation
- Resource allocation
- Risk identification
Use structured formats like bullet points and tables.`,
    temperature: 0.4,
    maxTokens: 600,
  });

  router
    .setRouter(routerAgent, ["code", "writing", "analysis", "planning"])
    .addRoute("code", codeAgent)
    .addRoute("writing", writingAgent)
    .addRoute("analysis", analysisAgent)
    .addRoute("planning", planningAgent)
    .setDefaultRoute("analysis");

  return router;
}
