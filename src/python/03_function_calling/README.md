# Module 3: Function Calling / Tool Use

## What is Function Calling?

Function calling (also called "tool use") allows LLMs to invoke external functions, APIs, and tools to accomplish tasks they can't do alone. Instead of just generating text, the model can:

- **Perform precise calculations** (instead of approximating math)
- **Access real-time data** (weather, stock prices, databases)
- **Take actions** (send emails, create files, update records)
- **Chain multiple tools** to complete complex tasks

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TOOL USE LOOP                                   │
│                                                                         │
│  ┌──────────┐    ┌──────────────────┐    ┌────────────────┐            │
│  │  User    │───>│   LLM decides    │───>│  Execute Tool  │            │
│  │  Query   │    │   which tool(s)  │    │  (your code)   │            │
│  └──────────┘    └──────────────────┘    └───────┬────────┘            │
│                          ↑                       │                      │
│                          │                       ↓                      │
│                  ┌───────┴─────────────────────────────┐               │
│                  │         Tool Result                  │               │
│                  │   (fed back to conversation)         │               │
│                  └───────────────────┬─────────────────┘               │
│                                      │                                  │
│                                      ↓                                  │
│                          ┌──────────────────┐                          │
│                          │   LLM decides:   │                          │
│                          │   More tools? or │                          │
│                          │   Final answer?  │                          │
│                          └────────┬─────────┘                          │
│                                   │                                     │
│                    ┌──────────────┴──────────────┐                     │
│                    │              │              │                      │
│               More tools     Final answer    Error                     │
│               (loop back)    (return to user)                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Three Tools Demonstrated

### 1. Calculator

**Purpose:** Precise mathematical operations

**Why needed:** LLMs can make arithmetic errors, especially with large numbers or complex operations.

```python
# Tool schema
{
    "name": "calculator",
    "input_schema": {
        "properties": {
            "operation": {"enum": ["add", "subtract", "multiply", "divide", "sqrt", ...]},
            "a": {"type": "number"},
            "b": {"type": "number"}
        }
    }
}

# Example call
calculator(operation="multiply", a=25, b=17)
# Returns: {"result": 425}
```

### 2. Weather API

**Purpose:** Access real-time weather data

**Why needed:** LLMs don't have access to current information after their training cutoff.

```python
# Tool schema
{
    "name": "get_weather",
    "input_schema": {
        "properties": {
            "city": {"type": "string"},
            "units": {"enum": ["celsius", "fahrenheit"]}
        }
    }
}

# Example call
get_weather(city="Tokyo", units="celsius")
# Returns: {"temperature": "28°C", "condition": "Sunny", "humidity": "70%"}
```

### 3. Product Database

**Purpose:** Query structured data

**Why needed:** LLMs can't access your specific databases or real-time inventory.

```python
# Tool schema
{
    "name": "query_products",
    "input_schema": {
        "properties": {
            "category": {"enum": ["electronics", "sports", "kitchen"]},
            "min_price": {"type": "number"},
            "max_price": {"type": "number"},
            "in_stock": {"type": "boolean"}
        }
    }
}

# Example call
query_products(category="electronics", max_price=50, in_stock=True)
# Returns: {"count": 1, "products": [{"name": "USB-C Cable", "price": 12.99, ...}]}
```

---

## Actual Results (Reproducible)

Run `python demo.py --provider anthropic` to reproduce:

### With Tools (100% Success)

| Query | Tools Called | Turns | Result |
|-------|-------------|-------|--------|
| "25 multiplied by 17" | calculator | 2 | **425** (correct) |
| "Weather in Tokyo" | get_weather | 2 | **28°C, Sunny** |
| "Electronics under $50" | query_products | 2 | **USB-C Cable $12.99** |
| "√144 then double it" | calculator, calculator | 3 | **24** (correct) |
| "Sports equipment + SF weather" | query_products, get_weather | 3 | **Yoga Mat, Water Bottle + 16°C Foggy** |

### Without Tools (0% Success)

| Query | Result |
|-------|--------|
| "25 multiplied by 17" | LLM computed correctly (425) but may fail on harder math |
| "Weather in Tokyo" | "I don't have access to real-time weather data" |
| "Electronics under $50" | "I don't have access to a database" |

---

## Multi-Step Tool Chains

The most powerful aspect of function calling is **chaining tools together**:

### Example: "√144 then double it"

```
Turn 1:
  User: "What's the square root of 144, and then double that result?"
  LLM: [calls calculator(operation="sqrt", a=144)]

Turn 2:
  Tool result: {"result": 12.0}
  LLM: [calls calculator(operation="multiply", a=12, b=2)]

Turn 3:
  Tool result: {"result": 24}
  LLM: "The square root of 144 is 12, and doubled that's 24."
```

### Example: "Sports equipment + weather advice"

```
Turn 1:
  User: "What sports equipment is under $50? Also, should I exercise outdoors in SF?"
  LLM: [calls query_products(category="sports", max_price=50)]

Turn 2:
  Tool result: {"products": [Yoga Mat $29.99, Water Bottle $19.99]}
  LLM: [calls get_weather(city="San Francisco")]

Turn 3:
  Tool result: {"temperature": "16°C", "condition": "Foggy", "humidity": "85%"}
  LLM: "Available sports equipment: Yoga Mat ($29.99), Water Bottle ($19.99).
        San Francisco is foggy with 85% humidity - might be better to exercise indoors."
```

---

## The Tool Use API Pattern

### 1. Define Tools

```python
tools = [
    {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "input_schema": { ... }
    },
    {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "input_schema": { ... }
    }
]
```

### 2. Call API with Tools

```python
response = client.messages.create(
    model="claude-3-haiku-20240307",
    tools=tools,
    messages=[{"role": "user", "content": "What's 25 * 17?"}]
)
```

### 3. Handle Tool Calls

```python
for block in response.content:
    if block.type == "tool_use":
        # LLM wants to call a tool
        tool_name = block.name      # "calculator"
        tool_input = block.input    # {"operation": "multiply", "a": 25, "b": 17}
        tool_id = block.id          # Used to match result

        # Execute the tool
        result = execute_tool(tool_name, tool_input)

        # Send result back to LLM
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user", "content": [{
            "type": "tool_result",
            "tool_use_id": tool_id,
            "content": json.dumps(result)
        }]})
```

### 4. Loop Until Done

```python
while has_tool_calls:
    response = client.messages.create(...)
    # Process tool calls
    # If no tool calls, we have the final answer
```

---

## Key Insights

1. **LLM as Orchestrator** - The model decides WHICH tools to use and WHEN to call them. You provide capabilities; the LLM provides intelligence.

2. **Precision via Tools** - For calculations, always use a calculator tool. LLMs make arithmetic mistakes, especially with large numbers.

3. **Real-Time Access** - Tools bridge the gap between the LLM's training data and current information.

4. **Multi-Turn Conversations** - Complex tasks require multiple tool calls. Each result feeds back into the conversation.

5. **Schema Enforcement** - Tool schemas ensure the LLM provides correct parameters (just like Module 2's structured outputs).

---

## Running the Demo

```bash
cd /home/adam/LLMControl
PYTHONPATH=src/python python src/python/03_function_calling/demo.py --provider anthropic
```

---

## Files in This Module

| File | Description |
|------|-------------|
| `tools.py` | Tool schemas, implementations, and test queries |
| `executor.py` | Agentic tool execution loop |
| `demo.py` | Comparison demo (with vs without tools) |
| `README.md` | This documentation |

---

## Connection to Previous Modules

| Module | Concept | Connection |
|--------|---------|------------|
| **Module 1** | Prompt Engineering | Tool descriptions are prompts that guide tool selection |
| **Module 2** | Structured Outputs | Tool schemas ARE structured output schemas |
| **Module 3** | Function Calling | Combines both: structured inputs → execute → structured outputs |

---

## When to Use Function Calling

| Use Case | Tool Type | Example |
|----------|-----------|---------|
| Math operations | Calculator | Financial calculations, unit conversions |
| Real-time data | API calls | Weather, stock prices, news |
| Database queries | SQL/NoSQL | Product search, user lookup |
| File operations | System tools | Read/write files, search |
| External actions | Webhooks | Send email, create ticket, post message |

---

## Comparison: With vs Without Tools

| Aspect | Without Tools | With Tools |
|--------|---------------|------------|
| **Math accuracy** | Can make errors | 100% precise |
| **Real-time data** | "I don't have access" | Actual current data |
| **Database queries** | "I can't access your DB" | Actual query results |
| **Latency** | ~1 second | ~3 seconds (multiple turns) |
| **Cost** | Lower (single turn) | Higher (multiple turns) |

---

## Next Steps

After mastering function calling, move to:
- **Module 4: RAG** - Augment LLM knowledge with retrieved documents
- **Module 5: Guardrails** - Add safety layers around tool use
