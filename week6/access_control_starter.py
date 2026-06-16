import json
import logging
import re
from typing import Dict, Any, List
from datetime import datetime
from time import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AccessController:
    def __init__(self, access_policy_path: str):
        with open(access_policy_path) as f:
            self.policy = json.load(f)
        self.audit_log = []

    def can_view_document(self, role: str, document: Dict[str, Any]) -> bool:
        sensitivity = document.get("sensitivity", "Public")
        allowed_roles = self.policy["document_access"].get(sensitivity, [])
        return role in allowed_roles

    def can_view_field(self, role: str, field_name: str) -> bool:
        field_policy = self.policy["sensitive_fields"].get(field_name)
        if not field_policy:
            return True
        return role in field_policy["visibility"]

    def redact_response(self, role: str, response: str) -> str:
        for field, field_policy in self.policy["sensitive_fields"].items():
            if role not in field_policy["visibility"]:
                patterns = [
                    rf'(?i)("{field}"\s*:\s*)"([^"]*)"',
                    rf'(?i)("{field}"\s*:\s*)(\d[\d,\.]*)',
                    rf'(?i)(\b{field}\b\s*(?:is|:)\s*\$?)([\d,\.]+)',
                    rf'(?i)(\b{field}\b\s*(?:is|:)\s*)([^\n]+)',
                    ]
                for pattern in patterns:
                    response = re.sub(pattern, r'\1[REDACTED]', response)
        return response

    def log_access(self, role: str, resource: str, allowed: bool, field: str = None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "resource": resource,
            "field": field,
            "allowed": allowed,
        }
        self.audit_log.append(entry)
        logger.info(f"Access {'ALLOWED' if allowed else 'DENIED'} - role={role} resource={resource} field={field}")

    def filter_documents(self, role: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for doc in documents:
            allowed = self.can_view_document(role, doc)
            self.log_access(role, doc.get("id", "unknown"), allowed)
            if allowed:
                result.append(doc)
        return result

    def get_audit_log(self) -> List[Dict[str, Any]]:
        return self.audit_log


class RateLimiter:
    def __init__(self, max_queries_per_minute: int = 30):
        self.max_queries_per_minute = max_queries_per_minute
        self.user_query_times = {}

    def is_allowed(self, user_id: str) -> bool:
        now = time()
        if user_id not in self.user_query_times:
            self.user_query_times[user_id] = []
        self.user_query_times[user_id] = [
            t for t in self.user_query_times[user_id] if now - t < 60
        ]
        if len(self.user_query_times[user_id]) < self.max_queries_per_minute:
            self.user_query_times[user_id].append(now)
            return True
        return False

    def get_remaining_queries(self, user_id: str) -> int:
        now = time()
        if user_id not in self.user_query_times:
            return self.max_queries_per_minute
        recent = [t for t in self.user_query_times[user_id] if now - t < 60]
        return max(0, self.max_queries_per_minute - len(recent))


class CostEnforcer:
    def __init__(self, policy_path: str = None):
        self.role_budgets = {
            "engineer": 100.0,
            "manager": 500.0,
            "hr": 200.0,
            "finance": 500.0,
            "executive": 1000.0,
        }
        self.user_spending = {}

    def add_cost(self, user_id: str, role: str, cost: float):
        if user_id not in self.user_spending:
            self.user_spending[user_id] = {"role": role, "total": 0.0}
        self.user_spending[user_id]["total"] += cost

    def can_afford_query(self, user_id: str, estimated_cost: float, role: str = "engineer") -> bool:
        if user_id not in self.user_spending:
            budget = self.role_budgets.get(role, 100.0)
            return estimated_cost <= budget
        user_role = self.user_spending[user_id]["role"]
        budget = self.role_budgets.get(user_role, 100.0)
        spent = self.user_spending[user_id]["total"]
        return estimated_cost <= (budget - spent)

    def get_budget_remaining(self, user_id: str) -> float:
        if user_id not in self.user_spending:
            return 0.0
        user_role = self.user_spending[user_id]["role"]
        budget = self.role_budgets.get(user_role, 100.0)
        spent = self.user_spending[user_id]["total"]
        return max(0.0, budget - spent)


if __name__ == "__main__":
    print("Testing AccessController...")
    controller = AccessController("data/access_control.json")

    assert not controller.can_view_field("engineer", "salary"), "Engineer should not see salary"
    assert controller.can_view_field("hr", "salary"), "HR should see salary"
    assert controller.can_view_field("manager", "salary"), "Manager should see salary"
    assert not controller.can_view_field("engineer", "ssn"), "Engineer should not see SSN"
    print("  can_view_field: PASSED")

    docs = [
        {"id": "doc1", "sensitivity": "Public", "content": "Mission statement"},
        {"id": "doc2", "sensitivity": "Confidential", "content": "Salary ranges"},
    ]
    visible = controller.filter_documents("engineer", docs)
    assert len(visible) == 1 and visible[0]["id"] == "doc1", "Engineer should only see Public doc"
    print("  filter_documents: PASSED")

    print("\nTesting RateLimiter...")
    limiter = RateLimiter(max_queries_per_minute=3)
    assert limiter.is_allowed("user1"), "First query should be allowed"
    assert limiter.is_allowed("user1"), "Second query should be allowed"
    assert limiter.is_allowed("user1"), "Third query should be allowed"
    assert not limiter.is_allowed("user1"), "Fourth query should be blocked"
    print("  is_allowed: PASSED")

    print("\nTesting CostEnforcer...")
    enforcer = CostEnforcer()
    assert enforcer.can_afford_query("user1", 50.0, role="engineer"), "Should afford $50 within $100 budget"
    enforcer.add_cost("user1", "engineer", 50.0)
    assert enforcer.can_afford_query("user1", 49.0), "Should afford $49 with $50 remaining"
    assert not enforcer.can_afford_query("user1", 51.0), "Should not afford $51 with $50 remaining"
    print("  can_afford_query: PASSED")

    print("\nAll tests passed!")
