import json
import sqlite3
from typing import Dict, Any
from groq import Groq
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


class Tool:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def execute(self, **kwargs) -> str:
        raise NotImplementedError


class EmployeeLookupTool(Tool):
    def __init__(self, db_path: str):
        super().__init__("employee_lookup", "Find employee information by name or ID")
        self.db_path = db_path

    def execute(self, employee_name: str = None, employee_id: str = None, name: str = None, id: str = None, **kwargs) -> str:
        employee_name = employee_name or name
        employee_id = employee_id or id
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if employee_id:
                cursor.execute("SELECT * FROM employees WHERE id = ?", (employee_id,))
            elif employee_name:
                cursor.execute("SELECT * FROM employees WHERE name LIKE ?", (f"%{employee_name}%",))
            else:
                return "Please provide either employee_name or employee_id"
            rows = cursor.fetchall()
            conn.close()
            if not rows:
                return "Employee not found"
            results = [dict(row) for row in rows]
            return json.dumps(results, indent=2)
        except Exception as e:
            logger.error(f"Employee lookup error: {e}")
            return f"Error: {str(e)}"


class PolicySearchTool(Tool):
    def __init__(self):
        super().__init__("policy_search", "Search policy documents by keyword or topic")
        try:
            with open("data/documents.json") as f:
                self.documents = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            self.documents = []

    def execute(self, query: str = None, topic: str = None, keyword: str = None, limit: int = 5, **kwargs) -> str:
        try:
            search_term = (query or topic or keyword or "").replace("_", " ").replace("-", " ")
            matches = [doc for doc in self.documents if search_term.lower() in doc["content"].lower() or search_term.lower() in doc["title"].lower()]
            if not matches:
                return f"No documents found matching: {search_term}"
            results = []
            for doc in matches[:limit]:
                snippet = doc["content"][:500].strip()
                results.append(f"Title: {doc['title']}\nSnippet: {snippet}\n")
            return "\n---\n".join(results)
        except Exception as e:
            logger.error(f"Policy search error: {e}")
            return f"Error: {str(e)}"


class ExpenseQueryTool(Tool):
    def __init__(self):
        super().__init__("expense_query", "Query expense approval limits by role")
        try:
            with open("data/policies.json") as f:
                self.policies = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load policies: {e}")
            self.policies = {}

    def execute(self, role: str = None, **kwargs) -> str:
        try:
            if not role:
                return "Please provide a role"
            limits = self.policies.get("expense", {}).get("approval_limits", {})
            if role in limits:
                amount = limits[role]
                return "Approval limit for " + role + ": $" + str(amount)
            return "Role not found: " + role + ". Valid roles: " + ", ".join(limits.keys())
        except Exception as e:
            logger.error(f"Expense query error: {e}")
            return f"Error: {str(e)}"


class Agent:
    def __init__(self, db_path: str, api_key: str = None):
        self.db_path = db_path
        self.api_key = api_key or GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set.")
        self.client = Groq(api_key=self.api_key)
        self.tools = {
            "employee_lookup": EmployeeLookupTool(db_path),
            "policy_search": PolicySearchTool(),
            "expense_query": ExpenseQueryTool(),
        }
        self.token_count = 0
        self.total_cost = 0.0
        self.queries_run = 0

    def _build_system_prompt(self, user_role: str) -> str:
        return (
            "You are a TechCorp internal assistant. Answer employee questions using the tools below.\n"
            "User role: " + user_role + "\n\n"
            "Available tools:\n"
            "- employee_lookup: Find employee information by name or ID\n"
            "- policy_search: Search policy documents by keyword or topic\n"
            "- expense_query: Query expense approval limits by role (ic1_ic2, ic3, manager, director, vp)\n\n"
            "To use a tool, respond with exactly this format:\n"
            "TOOL: <tool_name>\n"
            "ARGS: <argument>=<value>\n\n"
            "Only use one tool at a time. Be specific with arguments."
        )

    def _parse_tool_call(self, response_text: str):
        tool_name = None
        args = {}
        for line in response_text.strip().split("\n"):
            if line.startswith("TOOL:"):
                tool_name = line.replace("TOOL:", "").strip()
            elif line.startswith("ARGS:"):
                args_str = line.replace("ARGS:", "").strip()
                for part in args_str.split(","):
                    if "=" in part:
                        key, val = part.split("=", 1)
                        args[key.strip()] = val.strip()
        return tool_name, args

    def query(self, user_query: str, user_role: str = "engineer") -> Dict[str, Any]:
        logger.info(f"Processing query: {user_query}")
        total_tokens = 0

        system_prompt = self._build_system_prompt(user_role)

        response1 = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )

        response1_text = response1.choices[0].message.content
        total_tokens += response1.usage.total_tokens

        tool_name, args = self._parse_tool_call(response1_text)
        if tool_name and tool_name in self.tools:
            logger.info(f"Calling tool: {tool_name} with args: {args}")
            tool_result = self.tools[tool_name].execute(**args)

            response2 = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query},
                    {"role": "assistant", "content": response1_text},
                    {"role": "user", "content": "Tool result from " + tool_name + ":\n" + tool_result[:2000] + "\n\nNow answer the original question in plain English. Do not call any more tools. Just summarize the information for the user."}
                ]
            )
            final_answer = response2.choices[0].message.content
            total_tokens += response2.usage.total_tokens
        else:
            final_answer = response1_text

        cost = self._estimate_query_cost(total_tokens, 0)
        self.token_count += total_tokens
        self.total_cost += cost
        self.queries_run += 1

        return {
            "answer": final_answer,
            "tokens_used": total_tokens,
            "cost": cost,
            "role": user_role,
        }

    def _estimate_query_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens / 1_000_000) * 0.075

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_queries": self.queries_run,
            "total_tokens": self.token_count,
            "total_cost": self.total_cost,
            "avg_cost_per_query": self.total_cost / self.queries_run if self.queries_run > 0 else 0.0,
        }


if __name__ == "__main__":
    import sys
    try:
        agent = Agent("data/techcorp.db")
        print("Agent initialized successfully")

        print("\nTesting query: What is the travel policy?")
        result = agent.query("What is the travel policy?")
        print("Answer:", result["answer"])
        print("Tokens:", result["tokens_used"])
        print("Cost: $" + str(round(result["cost"], 6)))

        metrics = agent.get_metrics()
        print("\nMetrics:", metrics)

    except Exception as e:
        print(f"Error: {e}")
        logger.exception("Error during test")
        sys.exit(1)
