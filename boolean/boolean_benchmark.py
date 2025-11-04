import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
import ast
import sympy as sp
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from textwrap import dedent
from collections import Counter
from scipy import stats
import traceback

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent))

from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM
from boolean_dataset import BooleanExpression, BooleanObservation

# 添加 DeepSeekLLM 类
from openai import OpenAI

class DeepSeekLLM(LLMInterface):
    def __init__(self, model: str, api_key: str, temperature: float = 0.7):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1"  # DeepSeek API 端点
        )
        self.temperature = temperature
    
    def query(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error querying DeepSeek: {str(e)}"
    
    def get_name(self) -> str:
        return f"DeepSeek({self.model})"
    
    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """查询并返回使用情况"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=500
            )
            
            # 提取使用情况
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            total_tokens = usage.total_tokens if usage else 0
            
            return {
                'response': response.choices[0].message.content,
                'usage': {
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'total_tokens': total_tokens
                },
                'cost': 0.0  # DeepSeek 成本计算需要根据实际定价
            }
        except Exception as e:
            return {
                'response': f"Error querying DeepSeek: {str(e)}",
                'usage': {
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'total_tokens': 0
                },
                'cost': 0.0
            }


class BooleanBenchmarkRefined:
    def __init__(self, complete_dataset_path: str):
        """
        Initialize benchmark with a complete dataset.
        
        Args:
            complete_dataset_path: Path to the complete Boolean dataset JSON file
        """
        with open(complete_dataset_path, 'r') as f:
            self.complete_dataset = json.load(f)
        
        # Extract metadata
        self.mechanistic_opts = self.complete_dataset['metadata']['mechanistic_opts']
        self.metadata = self.complete_dataset['metadata']
        self.variables = self.metadata['variables']
        self.operators = set(self.metadata['operators'])
        self.max_depth = self.metadata['max_depth']
        
        # Flatten all datasets into a single list for sampling
        self.all_observation_sets = []
        for n_obs, datasets in self.complete_dataset['datasets_by_n_observations'].items():
            self.all_observation_sets.extend(datasets)
        
        print(f"Loaded complete dataset with {len(self.all_observation_sets)} observation sets")
        print(f"Variables: {', '.join(self.variables)}")
        print(f"Operators: {', '.join(self.operators)}")
        print(f"Max depth: {self.max_depth}")
    
    def sample_observation_sets(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        """
        Sample n observation sets from the complete dataset.
        
        Args:
            n_samples: Number of observation sets to sample
            seed: Random seed for reproducibility
        
        Returns:
            List of sampled observation sets
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Sample without replacement
        n_available = len(self.all_observation_sets)
        n_to_sample = min(n_samples, n_available)
        
        if n_to_sample < n_samples:
            print(f"Warning: Requested {n_samples} samples but only {n_available} available")
        
        sampled = random.sample(self.all_observation_sets, n_to_sample)
        return sampled

    def create_prompt(self, observations: List[BooleanObservation], 
                     prior_hypotheses: List[str], enable_cot: bool = True) -> str:
        # Build obs_block, prior_block, operators_str ...
        # Format observations
        obs_lines = []
        for obs in observations:
            obs_lines.append(obs.to_string())
        obs_block = "\n".join(obs_lines)
        
        # Format prior hypotheses
        if prior_hypotheses:
            prior_block = "\n".join([f"Expression: {h}" for h in prior_hypotheses])
        else:
            prior_block = "None"
        
        # Build operator description
        op_descriptions = []
        if 'AND' in self.operators: op_descriptions.append("AND (conjunction)")
        if 'OR'  in self.operators: op_descriptions.append("OR (disjunction)")
        if 'NOT' in self.operators: op_descriptions.append("NOT (negation)")
        if 'XOR' in self.operators: op_descriptions.append("XOR (exclusive or)")
        if 'NOR' in self.operators: op_descriptions.append("NOR (NOT (x OR y))")
        
        operators_str = ", ".join(op_descriptions)

        # Pull mechanistic opts from the loaded dataset
        mech = self.complete_dataset.get('metadata', {}).get('mechanistic_opts', {})
        comm = mech.get('apply_commutativity', True)
        idem = mech.get('apply_idempotence_and_or', True)
        flat = mech.get('flatten_associativity', True)

        eq_rules = []
        if comm:
            eq_rules.append(
                "- Commutativity: reorderings are the SAME (e.g., x AND y = y AND x)."
            )
        if idem:
            eq_rules.append(
                "- Idempotence: duplicates of the same input under AND/OR collapse (e.g., x AND x = x; x OR x = x)."
            )
        if flat:
            eq_rules.append(
                "- Associativity flattening: cascades of the SAME operator are the SAME regardless of parentheses "
                "(e.g., (x AND y) AND x = x AND (y AND x) = x AND x AND y)."
            )
        else:
            eq_rules.append(
                "- No associativity flattening: different parenthesizations of the SAME operator are DIFFERENT "
                "(e.g., (x AND y) AND x ≠ x AND (y AND x))."
            )

        eq_rules.append(
            "- No distributivity, absorption, or De Morgan normalization: mixing operators keeps expressions DIFFERENT "
            "(e.g., (x AND y) OR z ≠ x AND (y OR z))."
        )

        rules_text = "\n        ".join(eq_rules)

        # Enhanced reasoning guidance based on your data patterns
        reasoning_guidance = """
        REASONING STRATEGY FOR BOOLEAN DISCOVERY:

        1. PATTERN ANALYSIS PHASE:
           - Group observations by output (True=1 vs False=0)
           - Count how many True outputs you have
           - Look for correlations with single variables
           * If output matches one variable exactly → Use that variable
           * If output is opposite of one variable → Use NOT on that variable
           * If output is 1 when either variable is 1 → Use OR
           * If output is 1 only when both variables are 1 → Use AND

        2. SYSTEMATIC TESTING APPROACH:
           Step 1: Test single variables first:
             - Check if output = x
             - Check if output = y
             - Check if output = NOT x  
             - Check if output = NOT y

           Step 2: Test 2-variable combinations:
             - Check if output = x OR y
             - Check if output = x AND y
             - Check if output = NOT (x AND y)  # NAND
             - Check if output = NOT (x OR y)   # NOR

        3. VERIFICATION CHECK:
           - MUST work for ALL given observations
           - Test each candidate mentally against all input combinations
           - Eliminate any expression that fails even one observation

        4. SIMPLICITY PRINCIPLE:
           - Prefer single-variable expressions over complex ones
           - Choose the expression with fewest operators
           - Avoid unnecessary parentheses when possible
        """

        # Examples based on your provided data patterns
        examples = """
        EXAMPLE PATTERNS:

        Example 1 - OR Pattern:
        Observations:
        (x=0, y=0) -> 0
        (x=0, y=1) -> 1  
        (x=1, y=0) -> 1
        (x=1, y=1) -> 1
        Reasoning: Output is 0 only when both inputs are 0, matches OR behavior
        Final: Expression: x OR y

        Example 2 - NOT Pattern (NOT y):
        Observations:
        (x=0, y=0) -> 1
        (x=0, y=1) -> 0
        (x=1, y=0) -> 1  
        (x=1, y=1) -> 0
        Reasoning: Output depends only on y: when y=0→1, when y=1→0
        Final: Expression: NOT y

        Example 3 - NOT Pattern (NOT x):
        Observations:
        (x=0, y=0) -> 1
        (x=0, y=1) -> 1
        (x=1, y=0) -> 0
        (x=1, y=1) -> 0
        Reasoning: Output depends only on x: when x=0→1, when x=1→0
        Final: Expression: NOT x

        Example 4 - AND Pattern:
        Observations:
        (x=0, y=0) -> 0
        (x=0, y=1) -> 0
        (x=1, y=0) -> 0
        (x=1, y=1) -> 1
        Reasoning: Output is 1 only when both inputs are 1, matches AND behavior
        Final: Expression: x AND y
        """

        # 添加思维链推理块
        cot_block = ""
        if enable_cot:
            cot_block = """
            Internal Deliberation (do not reveal):
            - For each input combination, compute the expected output based on your candidate expression.
            - Ensure the computed output matches ALL given observations exactly.
            - If any mismatch exists, adjust the expression (change operators, variables, or structure) until all observations match.
            - Do NOT output your reasoning or steps. Only output the final expression line.
            """.strip()

        prompt = f"""You are given partial observations of a Boolean function with variables: {', '.join(self.variables)}

{reasoning_guidance}

{examples}

{cot_block}

Allowed operators: {operators_str}

Observations (input -> output):
{obs_block}

Prior expressions generated (avoid repeating any expression that is equivalent under the rules below):
{prior_block}

Task: Generate a single Boolean expression that is consistent with ALL observations.

Requirements:
1. Use ONLY the variables: {', '.join(self.variables)}
2. Use ONLY these operators: {', '.join(self.operators)}
3. The expression must match all given observations
4. Structural uniqueness is judged by these rules:
{rules_text}
5. Expression depth should be at most {self.max_depth} levels of nesting
6. Do not use boolean constants True or False anywhere in the expression.

THINKING PROCESS:
Follow the reasoning strategy above to systematically analyze the pattern. Test candidate expressions and verify they match ALL observations.

Output format: 
- Return ONLY the Boolean expression on a single line
- Use plain text format (no LaTeX, no markdown, no special formatting)
- Use uppercase for operators. 
- Use lowercase for variables: x, y
- Use parentheses for grouping when needed
- Start your response with "Expression: " followed by the expression

Examples of correct format:
Expression: x AND y
Expression: (x OR y) AND NOT x
Expression: NOT (x AND y)
Expression: NOR(x, y)

DO NOT use formats like:
- \\(x \\land y\\)  (LaTeX)
- `x AND y`  (markdown)
- x ∧ y  (mathematical symbols)

After your analysis, provide your final answer in this exact format:
Expression: [your boolean expression]
"""
        return prompt
    
    def parse_llm_response(self, response: str) -> Optional[str]:
        """Parse LLM response to extract Boolean expression."""
        if not isinstance(response, str):
            return None
        
        response = response.strip()
        
        # Clean common formatting issues
        def clean_expression(expr: str) -> str:
            # Remove LaTeX delimiters
            expr = re.sub(r'\\[()\[\]]', '', expr)
            expr = re.sub(r'\$+', '', expr)
            
            # Remove markdown backticks
            expr = expr.strip('`')
            
            # Convert LaTeX/math symbols to text
            expr = expr.replace('\\land', 'AND')
            expr = expr.replace('\\lor', 'OR')
            expr = expr.replace('\\lnot', 'NOT')
            expr = expr.replace('\\neg', 'NOT')
            expr = expr.replace('\\wedge', 'AND')
            expr = expr.replace('\\vee', 'OR')
            expr = expr.replace('∧', 'AND')
            expr = expr.replace('∨', 'OR')
            expr = expr.replace('¬', 'NOT')
            expr = expr.replace('⊕', 'XOR')
            
            # Remove \text{} wrappers
            expr = re.sub(r'\\text\{([^}]+)\}', r'\1', expr)
            
            # Clean up extra spaces and symbols
            expr = re.sub(r'\*+$', '', expr)  # Remove trailing asterisks
            expr = expr.strip('"\'').rstrip('.,;')
            
            return expr.strip()
        
        # Look for "Expression:" line
        lines = response.split('\n')
        for line in lines:
            if 'Expression:' in line or 'expression:' in line:
                parts = re.split(r'[Ee]xpression:', line)
                if len(parts) >= 2:
                    expr = clean_expression(parts[1])
                    if expr:
                        return expr
        
        # Fallback: try first non-empty line with operators
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                cleaned = clean_expression(line)
                if any(op in cleaned.upper() for op in ['AND','OR','NOT','XOR','NOR']):
                    return cleaned
                if all(c.isalnum() or c in '() ' for c in cleaned):
                    return cleaned
        
        return None

    def get_expression_mechanistic_key(self, expr_str: str) -> Optional[Tuple]:
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            return expr.mechanistic_key(**self.mechanistic_opts)
        except:
            return None
    
    def _in_space(self, expr_str: str) -> bool:
        """Check if expression meets all space constraints (variables, operators, depth, no constants)."""
        try:
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            
            if expr.sympy_expr is None:
                return False
            
            # Check variables
            allowed_symbols = {sp.symbols(v) for v in self.variables}
            if not expr.sympy_expr.free_symbols.issubset(allowed_symbols):
                return False
            
            # Check operators
            if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
                return False
            
            # Check no constants
            for node in sp.preorder_traversal(expr.sympy_expr):
                if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                    return False
            
            # Check depth
            if self._get_ast_depth(expr.sympy_expr) > self.max_depth:
                return False
            
            return True
        except:
            return False
    
    def _get_ast_depth(self, sympy_expr) -> int:
        """Calculate the true AST depth of an expression."""
        if sympy_expr is None:
            return 0
        
        # Leaf nodes (variables, constants) have depth 0
        if isinstance(sympy_expr, sp.Symbol):
            return 0
        if isinstance(sympy_expr, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
            return 0
        
        # For operators, depth is 1 + max depth of children
        if hasattr(sympy_expr, 'args') and len(sympy_expr.args) > 0:
            child_depths = [self._get_ast_depth(arg) for arg in sympy_expr.args]
            return 1 + max(child_depths)
        
        return 0
    
    def _check_operators_in_ast(self, sympy_expr, allowed_ops: Set[str]) -> bool:
        if sympy_expr is None:
            return False
        stack = [sympy_expr]
        while stack:
            node = stack.pop()

            # NOR pattern: Not(Or(...))
            if isinstance(node, sp.Not) and isinstance(node.args[0], sp.Or):
                if 'NOR' not in allowed_ops:
                    return False
                # treat as primitive; do NOT traverse inside
                continue

            # primitive checks
            if isinstance(node, sp.Not) and 'NOT' not in allowed_ops:
                return False
            elif isinstance(node, sp.And) and 'AND' not in allowed_ops:
                return False
            elif isinstance(node, sp.Or) and 'OR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Xor) and 'XOR' not in allowed_ops:
                return False
            elif isinstance(node, sp.Implies) and 'IMPLIES' not in allowed_ops:
                return False
            elif isinstance(node, sp.Equivalent) and 'EQUIV' not in allowed_ops:
                return False

            if hasattr(node, 'args'):
                stack.extend(node.args)
        return True
    
    def validate_expression(self, expr_str: str, observations: List[BooleanObservation]) -> Tuple[bool, Optional[Dict]]:
        """Validate if expression is consistent with observations and meets constraints."""
        try:
            # Now create expression and validate
            expr = BooleanExpression(expr_str, self.variables, self.operators)
            
            # Check if expression uses only allowed variables
            if expr.sympy_expr is not None:
                allowed_symbols = {sp.symbols(v) for v in self.variables}
                free_symbols = expr.sympy_expr.free_symbols
                if not free_symbols.issubset(allowed_symbols):
                    return False, None
                
                # Check operators using AST traversal
                if not self._check_operators_in_ast(expr.sympy_expr, self.operators):
                    return False, None
                
                # Disallow constants (True/False) to maintain consistent space
                for node in sp.preorder_traversal(expr.sympy_expr):
                    if isinstance(node, (sp.logic.boolalg.BooleanTrue, sp.logic.boolalg.BooleanFalse)):
                        return False, None
            
            # Check expression depth using AST
            if expr.sympy_expr is not None:
                ast_depth = self._get_ast_depth(expr.sympy_expr)
                if ast_depth > self.max_depth:
                    return False, None
            
            # Check consistency with observations
            for obs in observations:
                if expr.evaluate(obs.inputs) != obs.output:
                    return False, None
            
            return True, expr.truth_table
        except:
            return False, None
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error from the error message with more detail."""
        if "Expecting value" in error_message:
            # Extract line and column info if available
            match = re.search(r'line (\d+) column (\d+)', error_message)
            if match:
                return f"json_parse_error (line {match.group(1)}, col {match.group(2)})"
            return "json_parse_error"
        elif "Rate limit" in error_message.lower():
            return "rate_limit"
        elif "timeout" in error_message.lower():
            return "timeout"
        elif "401" in error_message or "unauthorized" in error_message.lower():
            return "auth_error"
        elif "403" in error_message or "forbidden" in error_message.lower():
            return "forbidden_error"
        elif "404" in error_message:
            return "not_found_error"
        elif "429" in error_message:
            return "rate_limit_429"
        elif "500" in error_message or "internal server error" in error_message.lower():
            return "server_error_500"
        elif "502" in error_message or "bad gateway" in error_message.lower():
            return "bad_gateway_502"
        elif "503" in error_message or "service unavailable" in error_message.lower():
            return "service_unavailable_503"
        elif "connection" in error_message.lower():
            return "connection_error"
        elif "JSONDecodeError" in error_message:
            return "json_decode_error"
        else:
            # Try to extract HTTP status code
            match = re.search(r'\b(\d{3})\b', error_message)
            if match:
                return f"http_error_{match.group(1)}"
            return "unknown_error"
    
    def get_expression_depth(self, expr_str: str) -> int:
        """Calculate the depth of a Boolean expression."""
        # Simple heuristic: count maximum nesting level of parentheses
        max_depth = 0
        current_depth = 0
        
        for char in expr_str:
            if char == '(':
                current_depth += 1
                max_depth = max(max_depth, current_depth)
            elif char == ')':
                current_depth -= 1
        
        # Also check for operators without parentheses
        if max_depth == 0:
            # Count operators to estimate depth
            expr_upper = expr_str.upper()
            if any(op in expr_upper for op in ['AND','OR','NOT','XOR','NOR']):
                max_depth = 1
        
        return max_depth
    
    def evaluate_single_observation_set(
        self,
        llm: LLMInterface,
        observation_set: Dict,
        n_queries: int = 10,
        verbose: bool = True,
        max_retries: int = 5
    ) -> Dict:
        """
        Evaluate LLM on a single observation set.
        
        Returns:
            Dictionary with evaluation results for this observation set
        """
        # Extract observations and ground truths
        observations = [
            BooleanObservation(obs['inputs'], obs['output']) 
            for obs in observation_set['observations']
        ]
        
        ground_truth_expressions = observation_set['ground_truth_expressions']
        gt_truth_tables = []
        gt_mechanistic_keys = []
        for gt in ground_truth_expressions:
            truth_table = {}
            for key_str, value in gt['truth_table'].items():
                key_tuple = ast.literal_eval(key_str)
                truth_table[key_tuple] = value
            gt_truth_tables.append(truth_table)
            # Extract mechanistic key for structural comparison
            gt_mech_key = gt.get('mechanistic_key', None)
            if gt_mech_key:
                # Convert string representation back to tuple if needed (safe parsing)
                gt_mechanistic_keys.append(ast.literal_eval(gt_mech_key) if isinstance(gt_mech_key, str) else gt_mech_key)
        
        # Track results
        all_hypotheses = []
        valid_hypotheses = []
        unique_mechanistic_keys = set()
        unique_valid_expressions = []
        all_unique_mechanistic_keys = set()
        unique_all_expressions = []
        parse_success_count = 0
        in_space_count = 0
        
        # Track token usage and costs
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Track errors
        errors = []  # List of error details
        error_counts = {}  # Count of each error type
        
        for i in range(n_queries):
            prompt = self.create_prompt(observations, all_hypotheses)
            
            # Try to get a valid response
            hypothesis_str = None
            query_error = None
            for attempt in range(max_retries):
                # Use query_with_usage if available
                if hasattr(llm, 'query_with_usage'):
                    result = llm.query_with_usage(prompt)
                    response = result['response']
                    # Track usage
                    total_prompt_tokens += result['usage']['prompt_tokens']
                    total_completion_tokens += result['usage']['completion_tokens']
                    total_tokens += result['usage']['total_tokens']
                    total_cost += result.get('cost', 0.0)
                else:
                    response = llm.query(prompt)
                
                # Check if response is an error
                if response.startswith("Error querying"):
                    query_error = {
                        'query_index': i,
                        'attempt': attempt + 1,
                        'error_message': response,
                        'error_type': self._classify_error(response)
                    }
                    # Track error type count
                    error_type = query_error['error_type']
                    error_counts[error_type] = error_counts.get(error_type, 0) + 1
                    continue  # Try again
                
                hypothesis_str = self.parse_llm_response(response)
                if hypothesis_str:
                    break
            
            # If all attempts failed, record the error
            if not hypothesis_str and query_error:
                errors.append(query_error)
                # Store detailed error info in all_hypotheses
                error_detail = f"[ERROR after {max_retries} attempts] Type: {query_error['error_type']} | {query_error['error_message']}"
                all_hypotheses.append(error_detail)
            
            elif hypothesis_str:  # Only if we got a valid hypothesis
                parse_success_count += 1
                all_hypotheses.append(hypothesis_str)
                
                # Only count novelty for in-space expressions
                if self._in_space(hypothesis_str):
                    in_space_count += 1
                    
                    # Check uniqueness among in-space hypotheses (for novelty calculation)
                    all_mech_key = self.get_expression_mechanistic_key(hypothesis_str)
                    if all_mech_key and all_mech_key not in all_unique_mechanistic_keys:
                        all_unique_mechanistic_keys.add(all_mech_key)
                        unique_all_expressions.append(hypothesis_str)
                
                # Validate expression against observations
                is_valid, truth_table = self.validate_expression(hypothesis_str, observations)
                
                if is_valid:
                    valid_hypotheses.append(hypothesis_str)
                    
                    # Check uniqueness among valid hypotheses
                    mech_key = self.get_expression_mechanistic_key(hypothesis_str)
                    if mech_key and mech_key not in unique_mechanistic_keys:
                        unique_mechanistic_keys.add(mech_key)
                        unique_valid_expressions.append(hypothesis_str)
        
        # Calculate metrics
        parse_success_rate = parse_success_count / n_queries if n_queries > 0 else 0
        in_space_rate = in_space_count / n_queries if n_queries > 0 else 0
        valid_rate = len(valid_hypotheses) / n_queries if n_queries > 0 else 0
        novelty_rate = len(unique_all_expressions) / n_queries if n_queries > 0 else 0
        recovery_rate = 0
        
        # Check recovery against ground truths using mechanistic keys (structural matching)
        recovered_gts = set()
        recovered_gt_keys = set()
        
        # Collect mechanistic keys of generated expressions
        generated_mech_keys = set()
        for expr in unique_valid_expressions:
            mech_key = self.get_expression_mechanistic_key(expr)
            if mech_key:
                generated_mech_keys.add(mech_key)
        
        # Check which ground truths were recovered (structurally)
        for j, gt_key in enumerate(gt_mechanistic_keys):
            if gt_key in generated_mech_keys:
                recovered_gts.add(j)
                recovered_gt_keys.add(gt_key)
        
        recovery_rate = len(recovered_gts) / len(gt_mechanistic_keys) if gt_mechanistic_keys else 0
        
        return {
            'observation_set_id': observation_set['observation_set_id'],
            'n_observations': observation_set['n_observations'],
            'n_ground_truths': len(ground_truth_expressions),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique_valid': len(unique_valid_expressions),  # Unique among valid hypotheses
            'n_unique_all': len(unique_all_expressions),  # Unique among in-space hypotheses
            'n_recovered_gts': len(recovered_gts),
            'parse_success_rate': parse_success_rate,  # Diagnostic: how many parsed
            'in_space_rate': in_space_rate,  # Diagnostic: how many met space constraints
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,  # Now using unique in-space / n_queries
            'recovery_rate': recovery_rate,
            'all_hypotheses': all_hypotheses,
            'valid_hypotheses': valid_hypotheses,
            'unique_valid_expressions': unique_valid_expressions,
            'unique_all_expressions': unique_all_expressions,
            # Token usage and cost tracking
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            # Error tracking
            'errors': errors,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_counts
            }
        }

    def evaluate_single_observation_set_teacher_student(
        self,
        student_llm: LLMInterface,
        teacher_llm: LLMInterface,
        observation_set: Optional[Dict] = None,
        observations: Optional[List[Dict]] = None,
        sample_id: Optional[Any] = None,
        sample_name: Optional[str] = None,
        n_queries: int = 10,
        verbose: bool = True,
        max_retries: int = 5
    ) -> Dict:
        """
        Evaluate LLMs on a single observation set with a Teacher–Student refinement loop.
        Student generates -> Teacher critiques -> Student refines.
        """
        # 兼容两种参数名
        if observation_set is not None:
            observations_data = observation_set
        elif observations is not None:
            observations_data = observations
        else:
            raise ValueError("Either observation_set or observations must be provided")

        # === 基础数据提取 ===
        observations = [
            BooleanObservation(obs['inputs'], obs['output']) 
            for obs in observations_data['observations']
        ]
        
        ground_truth_expressions = observations_data['ground_truth_expressions']
        gt_truth_tables = []
        gt_mechanistic_keys = []
        for gt in ground_truth_expressions:
            truth_table = {}
            for key_str, value in gt['truth_table'].items():
                key_tuple = ast.literal_eval(key_str)
                truth_table[key_tuple] = value
            gt_truth_tables.append(truth_table)
            # Extract mechanistic key for structural comparison
            gt_mech_key = gt.get('mechanistic_key', None)
            if gt_mech_key:
                gt_mechanistic_keys.append(ast.literal_eval(gt_mech_key) if isinstance(gt_mech_key, str) else gt_mech_key)

        # === 结果追踪 ===
        all_hypotheses = []
        valid_hypotheses = []
        unique_mechanistic_keys = set()
        unique_valid_expressions = []
        all_unique_mechanistic_keys = set()
        unique_all_expressions = []
        parse_success_count = 0
        in_space_count = 0
        teacher_feedbacks = []
        
        # === Token 与错误统计 ===
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        errors = []
        error_counts = {}

        # === 主循环：Student → Teacher → Student ===
        for i in range(n_queries):
            if verbose:
                print(f"\nQuery {i+1}/{n_queries}")
                print(f"  Step 1: Student generating initial hypothesis...")

            # 确保正确提取观察数据
            obs_data = observations_data['observations'] if isinstance(observations_data, dict) and 'observations' in observations_data else observations_data
            # 提供空的prior_hypotheses列表
            prompt = self.create_prompt(observations=observations, prior_hypotheses=[])
            hypothesis_str = None
            query_error = None
            
            try:
                student_result = student_llm.query(prompt)
                if verbose:
                    print(f"  Step 2: Teacher reviewing hypothesis...")
            except Exception as e:
                query_error = {'query_index': i, 'error_message': str(e)}
                errors.append(query_error)
                print(f"  Error: Failed to get student response: {str(e)}")
                continue

            # --- Step 2: 老师审阅反馈 ---
            feedback_prompt = f"""
            You are an expert Boolean reasoning teacher reviewing a student's proposed expression.

            IMPORTANT SEMANTICS OF BOOLEAN OBSERVATIONS:
            - Each observation shows input values and the corresponding output
            - The expression must produce the exact output for each input combination

            Observations:
            {json.dumps([{'inputs': obs.inputs, 'output': obs.output} for obs in observations], indent=2)}

            Student's proposed expression:
            ```
            {student_result}
            ```

            Please analyze whether this expression explains the observations logically.
            IMPORTANT RULES FOR BOOLEAN ANALYSIS:
            1. Test the expression with each input combination from observations
            2. Check if the computed output matches the observed output exactly
            3. Look for inconsistencies, missing operators, or incorrect logic
            4. Consider simpler alternatives if they explain the same pattern
            5. Only suggest edits when there is clear evidence of inconsistency
            6. If the student's expression already correctly explains all observations, confirm it's correct
            
            Example reasoning:
            - If expression is "x AND y" but observation (x=1, y=0) gives output=1, this is inconsistent
            - If expression is "x OR y" but observation (x=0, y=0) gives output=0, this is consistent
            
            - Identify any inconsistencies, missing operators, or unnecessary complexity.
            - Suggest concrete edits to make it valid and parsimonious.
            - Keep your feedback short and actionable.
            """
            try:
                teacher_feedback = teacher_llm.query(feedback_prompt)
                if verbose:
                    print(f"  Step 3: Student refining hypothesis...")
            except Exception as e:
                teacher_feedback = f"(Teacher failed: {str(e)})"
                print(f"  Warning: Teacher feedback generation failed: {str(e)}")

            # --- Step 3: 学生根据反馈修订 ---
            refine_prompt = f"""
            You previously proposed this expression:
            {student_result}

            Your teacher provided this feedback:
            "{teacher_feedback}"

            Now produce ONE revised Boolean expression that addresses the feedback
            and fits all observations. Keep it simple and correct.

            Respond with exactly one line:
            Expression: [your revised boolean expression]
            """
            try:
                refined_output = student_llm.query(refine_prompt)
                if verbose:
                    print(f"  Step 4: Parsing and validating refined hypothesis...")
            except Exception as e:
                refined_output = student_result  # 若 refinement 失败则保留初稿
                print(f"  Warning: Student refinement failed, using initial hypothesis: {str(e)}")
                
            # 添加到teacher_feedbacks列表
            teacher_feedbacks.append({
                'query_index': i,
                'student_initial': student_result,
                'teacher_feedback': teacher_feedback,
                'student_refined': refined_output
            })

            # --- Step 4: 解析输出 ---
            hypothesis_str = self.parse_llm_response(refined_output)
            if hypothesis_str:
                parse_success_count += 1
                if verbose:
                    print(f"Parsing successful")
            else:
                query_error = {
                    'query_index': i,
                    'error_message': f'Parsing failed for output: {refined_output[:100]}'
                }
                errors.append(query_error)
                print(f"Parsing failed: {refined_output[:50]}...")
                continue

            # === 收集 ===
            all_hypotheses.append(hypothesis_str)

            # Only count novelty for in-space expressions
            if self._in_space(hypothesis_str):
                in_space_count += 1
                
                # Check uniqueness among in-space hypotheses (for novelty calculation)
                all_mech_key = self.get_expression_mechanistic_key(hypothesis_str)
                if all_mech_key and all_mech_key not in all_unique_mechanistic_keys:
                    all_unique_mechanistic_keys.add(all_mech_key)
                    unique_all_expressions.append(hypothesis_str)

            # 验证有效性
            is_valid, truth_table = self.validate_expression(hypothesis_str, observations)
            if is_valid:
                valid_hypotheses.append(hypothesis_str)
                mech_key = self.get_expression_mechanistic_key(hypothesis_str)
                if mech_key and mech_key not in unique_mechanistic_keys:
                    unique_mechanistic_keys.add(mech_key)
                    unique_valid_expressions.append(hypothesis_str)
                if verbose:
                    print(f"Hypothesis is valid")
            else:
                if verbose:
                    print(f"Hypothesis is invalid")

        # === 统计指标 ===
        parse_success_rate = parse_success_count / n_queries if n_queries > 0 else 0
        in_space_rate = in_space_count / n_queries if n_queries > 0 else 0
        valid_rate = len(valid_hypotheses) / n_queries if n_queries > 0 else 0
        novelty_rate = len(unique_all_expressions) / n_queries if n_queries > 0 else 0

        # === Recovery 计算 ===
        recovered_gts = set()
        recovered_gt_keys = set()
        
        # Collect mechanistic keys of generated expressions
        generated_mech_keys = set()
        for expr in unique_valid_expressions:
            mech_key = self.get_expression_mechanistic_key(expr)
            if mech_key:
                generated_mech_keys.add(mech_key)
        
        # Check which ground truths were recovered (structurally)
        for j, gt_key in enumerate(gt_mechanistic_keys):
            if gt_key in generated_mech_keys:
                recovered_gts.add(j)
                recovered_gt_keys.add(gt_key)
        
        recovery_rate = len(recovered_gts) / len(gt_mechanistic_keys) if gt_mechanistic_keys else 0

        # === 输出结构 ===
        return {
            'teacher_feedbacks': teacher_feedbacks,
            'observation_set_id': observations_data.get('observation_set_id', 'unknown'),
            'n_observations': len(observations),
            'n_ground_truths': len(ground_truth_expressions),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique_valid': len(unique_valid_expressions),
            'n_unique_all': len(unique_all_expressions),
            'n_recovered_gts': len(recovered_gts),
            'parse_success_count': parse_success_count,
            'parse_success_rate': parse_success_rate,
            'in_space_rate': in_space_rate,
            'valid_rate': valid_rate,
            'novelty_rate': novelty_rate,
            'recovery_rate': recovery_rate,
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens
            },
            'cost': total_cost,
            'teacher_student': True,
            'errors': errors,
            'error_summary': {
                'total_errors': len(errors),
                'error_types': error_counts
            },
            'all_hypotheses': all_hypotheses,
            'valid_hypotheses': valid_hypotheses,
            'unique_valid_expressions': unique_valid_expressions,
            'unique_all_expressions': unique_all_expressions
        }
    
    def run_benchmark(
        self,
        llm: LLMInterface,
        teacher_llm: Optional[LLMInterface] = None,
        n_samples: int = 10,
        n_queries_per_sample: Optional[int] = None,
        query_multiplier: float = 2.0,
        seed: Optional[int] = None,
        verbose: bool = True,
        checkpoint_dir: str = "checkpoints",
        max_retries: int = 3,
        run_id: Optional[str] = None
    ) -> Dict:
        """
        Run the benchmark with sampling, supporting both single-model and teacher-student modes.
        
        Args:
            llm: LLM interface to use (student model in teacher-student mode)
            teacher_llm: Optional teacher LLM interface for teacher-student mode
            n_samples: Number of observation sets to sample
            n_queries_per_sample: Fixed number of queries per observation set (if None, uses query_multiplier)
            query_multiplier: Multiplier for n_gt to determine queries (default 2.0 means 2x ground truths)
            seed: Random seed
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints
            max_retries: Maximum retries per query
        
        Returns:
            Dictionary with complete benchmark results
        """
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID and safe filename
        if not run_id:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Sanitize LLM name for filename
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_{safe_llm_name}_{run_id}.json"
        
        print(f"\nRunning refined Boolean benchmark")
        print(f"LLM: {llm.get_name()}")
        if teacher_llm:
            print(f"Teacher LLM: {teacher_llm.get_name()}")
            print(f"Mode: Teacher-Student")
        else:
            print(f"Mode: Single Model")
        print(f"Sampling {n_samples} observation sets")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x number of ground truths (adaptive)")
        print(f"Checkpoint file: {checkpoint_file}")
        print("-" * 50)
        
        # Sample observation sets
        sampled_sets = self.sample_observation_sets(n_samples, seed)
        
        # Initialize results tracking
        all_results = []
        valid_rates = []
        novelty_rates = []
        recovery_rates = []
        parse_success_rates = []
        
        # Initialize token and cost tracking
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Initialize error tracking
        all_errors = []
        total_error_counts = {}
        
        # Load existing checkpoint if it exists
        start_idx = 0
        if checkpoint_file.exists():
            print(f"Using Recovered checkpoint file: {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                all_results = checkpoint_data['results']
                start_idx = len(all_results)
                print(f"Resuming from checkpoint: {start_idx}/{n_samples} completed")

                # Recalculate rates from checkpoint
                for result in all_results:
                    valid_rates.append(result['valid_rate'])
                    novelty_rates.append(result['novelty_rate'])
                    recovery_rates.append(result['recovery_rate'])
                    parse_success_rates.append(result.get('parse_success_rate', 1.0))

                # RESTORE TOKEN AND COST TOTALS
                if 'total_token_usage' in checkpoint_data:
                    total_prompt_tokens = checkpoint_data['total_token_usage'].get('prompt_tokens', 0)
                    total_completion_tokens = checkpoint_data['total_token_usage'].get('completion_tokens', 0)
                    total_tokens = checkpoint_data['total_token_usage'].get('total_tokens', 0)
                if 'total_cost' in checkpoint_data:
                    total_cost = checkpoint_data['total_cost']
        
        # Process each sampled observation set
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]
            
            if verbose:
                print(f"\nSample {idx + 1}/{n_samples}")
                print(f"  Observation set ID: {obs_set['observation_set_id']}")
                print(f"  Number of observations: {obs_set['n_observations']}")
                print(f"  Number of ground truths: {obs_set['n_compatible_expressions']}")
            
            try:
                # Determine number of queries for this observation set
                if n_queries_per_sample is not None:
                    # Use fixed number
                    n_queries = n_queries_per_sample
                else:
                    # Adaptive: use multiplier of ground truths
                    n_gt = obs_set['n_compatible_expressions']
                    n_queries = max(1, int(n_gt * query_multiplier))
                    if verbose:
                        print(f"  Using {n_queries} queries ({query_multiplier}x {n_gt} ground truths)")
                
                # Evaluate on this observation set
                if teacher_llm:
                    # Use teacher-student mode
                    result = self.evaluate_single_observation_set_teacher_student(
                        student_llm=llm,
                        teacher_llm=teacher_llm,
                        observation_set=obs_set,
                        n_queries=n_queries,
                        verbose=False,
                        max_retries=max_retries
                    )
                else:
                    # Use single model mode
                    result = self.evaluate_single_observation_set(
                        llm, obs_set, n_queries, verbose=False, max_retries=max_retries
                    )
                
                all_results.append(result)
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])
                parse_success_rates.append(result.get('parse_success_rate', 1.0))
                
                # Aggregate token usage and costs
                if 'token_usage' in result:
                    total_prompt_tokens += result['token_usage']['prompt_tokens']
                    total_completion_tokens += result['token_usage']['completion_tokens']
                    total_tokens += result['token_usage']['total_tokens']
                if 'cost' in result:
                    total_cost += result['cost']
                
                # Aggregate errors
                if 'errors' in result and result['errors']:
                    all_errors.extend(result['errors'])
                    # Update error type counts
                    if 'error_summary' in result:
                        for error_type, count in result['error_summary']['error_types'].items():
                            total_error_counts[error_type] = total_error_counts.get(error_type, 0) + count
                
                if verbose:
                    print(f"  Parse success rate: {result.get('parse_success_rate', 0):.2%}")
                    print(f"  Valid rate: {result['valid_rate']:.2%}")
                    print(f"  Novelty rate: {result['novelty_rate']:.2%}")
                    print(f"  Recovery rate: {result['recovery_rate']:.2%}")
                    if result.get('cost', 0) > 0:
                        print(f"  Cost: ${result['cost']:.6f}")
                
                # Save checkpoint after each sample
                checkpoint_data = {
                    'run_id': run_id,
                    'llm_name': llm.get_name(),
                    'teacher_llm_name': teacher_llm.get_name() if teacher_llm else None,
                    'n_samples': n_samples,
                    'n_queries_per_sample': n_queries_per_sample,
                    'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                    'results': all_results,
                    'total_token_usage': {
                        'prompt_tokens': total_prompt_tokens,
                        'completion_tokens': total_completion_tokens,
                        'total_tokens': total_tokens
                    },
                    'total_cost': total_cost,
                    'total_errors': len(all_errors),
                    'error_types': total_error_counts
                }
                
                with open(checkpoint_file, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)

            except Exception as e:
                print(f"  Error processing sample {idx + 1}: {str(e)}")
                traceback.print_exc()
                continue
        
        # Calculate statistics
        def calculate_stats(rates):
            if not rates:
                return {'mean': 0, 'std': 0, 'var': 0, 'min': 0, 'max': 0}
            return {
                'mean': np.mean(rates),
                'std': np.std(rates),
                'var': np.var(rates),
                'min': np.min(rates),
                'max': np.max(rates)
            }
        
        # Calculate p-values (one-sample t-test against null hypothesis of 0)
        def calculate_p_value(rates):
            if not rates or len(rates) < 2:
                return None
            t_stat, p_val = stats.ttest_1samp(rates, 0)
            return p_val
        
        # Compile final results
        final_results = {
            'run_id': run_id,
            'llm_name': llm.get_name(),
            'teacher_llm_name': teacher_llm.get_name() if teacher_llm else None,
            'mode': 'teacher_student' if teacher_llm else 'single_model',
            'n_samples': len(all_results),
            'n_queries_per_sample': n_queries_per_sample,
            'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
            'query_mode': 'fixed' if n_queries_per_sample is not None else f'adaptive_{query_multiplier}x',
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'statistics': {
                'parse_success_rate': {
                    **calculate_stats(parse_success_rates),
                    'p_value': calculate_p_value(parse_success_rates)
                },
                'valid_rate': {
                    **calculate_stats(valid_rates),
                    'p_value': calculate_p_value(valid_rates)
                },
                'novelty_rate': {
                    **calculate_stats(novelty_rates),
                    'p_value': calculate_p_value(novelty_rates)
                },
                'recovery_rate': {
                    **calculate_stats(recovery_rates),
                    'p_value': calculate_p_value(recovery_rates)
                }
            },
            'token_usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'total_tokens': total_tokens,
                'avg_tokens_per_sample': total_tokens / len(all_results) if all_results else 0,
                'avg_tokens_per_query': total_tokens / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'cost': {
                'total_cost': total_cost,
                'avg_cost_per_sample': total_cost / len(all_results) if all_results else 0,
                'avg_cost_per_query': total_cost / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'error_summary': {
                'total_errors': len(all_errors),
                'error_types': total_error_counts,
                'error_rate': len(all_errors) / (len(all_results) * (n_queries_per_sample or 1)) if all_results else 0
            },
            'per_sample_results': all_results
        }
        
        # Print summary
        print("\n" + "=" * 50)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 50)
        print(f"Samples evaluated: {len(all_results)}/{n_samples}")
        print(f"Mode: {'Teacher-Student' if teacher_llm else 'Single Model'}")
        
        for metric_name, metric_key in [('Parse Success Rate', 'parse_success_rate'),
                                        ('Valid Rate', 'valid_rate'), 
                                        ('Novelty Rate', 'novelty_rate'), 
                                        ('Recovery Rate', 'recovery_rate')]:
            stats_dict = final_results['statistics'][metric_key]
            print(f"\n{metric_name}:")
            print(f"  Mean ± Std: {stats_dict['mean']:.3f} ± {stats_dict['std']:.3f}")
            print(f"  Variance: {stats_dict['var']:.3f}")
            print(f"  Range: [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
            if stats_dict['p_value'] is not None:
                print(f"  p-value: {stats_dict['p_value']:.4f}")
        
        # Print token usage and cost summary
        print(f"\nToken Usage:")
        print(f"  Total tokens: {final_results['token_usage']['total_tokens']:,}")
        print(f"  Prompt tokens: {final_results['token_usage']['prompt_tokens']:,}")
        print(f"  Completion tokens: {final_results['token_usage']['completion_tokens']:,}")
        print(f"  Avg tokens/sample: {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg tokens/query: {final_results['token_usage']['avg_tokens_per_query']:.1f}")
        
        print(f"\nCost:")
        print(f"  Total cost: ${final_results['cost']['total_cost']:.4f}")
        print(f"  Avg cost/sample: ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg cost/query: ${final_results['cost']['avg_cost_per_query']:.6f}")
        
        # Print error summary if there were errors
        if final_results['error_summary']['total_errors'] > 0:
            print(f"\nErrors:")
            print(f"  Total errors: {final_results['error_summary']['total_errors']}")
            print(f"  Error rate: {final_results['error_summary']['error_rate']:.2%}")
            if final_results['error_summary']['error_types']:
                print(f"  Error types:")
                for error_type, count in final_results['error_summary']['error_types'].items():
                    print(f"    - {error_type}: {count}")
        
        print("=" * 50)
        
        # Clean up checkpoint file after successful completion
        if checkpoint_file.exists():
            try:
                checkpoint_file.unlink()
                print(f"\nCleaned up checkpoint: {checkpoint_file}")
            except Exception:
                pass
        
        return final_results


def setup_llm(llm_type: str, **kwargs) -> LLMInterface:
    """Set up the LLM interface based on type."""
    
    if llm_type == "openai":
        api_key = kwargs.get('api_key') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        return OpenAILLM(
            model=kwargs.get('model', 'gpt-4'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    elif llm_type == "anthropic":
        api_key = kwargs.get('api_key') or os.environ.get('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("Anthropic API key required")
        
        return AnthropicLLM(
            model=kwargs.get('model', 'claude-3-opus-20240229'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    elif llm_type == "openrouter":
        api_key = kwargs.get('api_key') or os.environ.get('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("OpenRouter API key required")
        
        return OpenRouterLLM(
            model=kwargs.get('model', 'anthropic/claude-3.5-sonnet'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    elif llm_type == "deepseek":
        api_key = kwargs.get('api_key') or os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DeepSeek API key required")
        
        # 使用新的 DeepSeekLLM 类
        return DeepSeekLLM(
            model=kwargs.get('model', 'deepseek-reasoner'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )
    
    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Run Boolean discovery benchmark with Teacher–Student LLM pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--dataset", required=True, help="Path to Boolean dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of observation sets to sample")
    parser.add_argument("--n-queries", type=int, default=None, help="Number of Student–Teacher refinement rounds per sample (optional, will use dynamic calculation if not specified)")
    parser.add_argument("--query-multiplier", type=float, default=1.0, help="Multiplier for adaptive queries (n_queries = n_gt * multiplier)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--output", default=None, help="Output result path")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    args = parser.parse_args()

    # === 1️⃣ 载入配置文件 ===
    config = load_config(args.config)
    if not config:
        print("无效的配置文件，配置为空。")
        sys.exit(1)

    llm_cfg = config.get('llm', {})
    benchmark_cfg = config.get('benchmark', {})

    # === 2️⃣ 解析 Teacher / Student ===
    # 支持两种写法：
    # (1) 新式 teacher/student
    # (2) 旧式单模型

    # New logic: prioritize teacher/student
    if 'teacher' in llm_cfg and 'student' in llm_cfg:
        teacher_cfg = llm_cfg['teacher']
        student_cfg = llm_cfg['student']

        teacher_llm = setup_llm(
            teacher_cfg['type'],
            model=teacher_cfg.get('model'),
            api_key=teacher_cfg.get('api_key') or os.environ.get('DEEPSEEK_API_KEY'),
            temperature=teacher_cfg.get('temperature', 0.4)
        )
        student_llm = setup_llm(
            student_cfg['type'],
            model=student_cfg.get('model'),
            api_key=student_cfg.get('api_key') or os.environ.get('DEEPSEEK_API_KEY'),
            temperature=student_cfg.get('temperature', 0.7)
        )
        print(f" Loaded Teacher model: {teacher_cfg['model']} ({teacher_cfg['type']})")
        print(f" Loaded Student model: {student_cfg['model']} ({student_cfg['type']})")
        mode = "teacher_student"
    else:
        # 旧兼容逻辑（单模型）
        llm_type = llm_cfg.get('type', 'openrouter')
        model = llm_cfg.get('models', {}).get(llm_type)
        api_key = llm_cfg.get('api_keys', {}).get(llm_type)
        temperature = llm_cfg.get('temperature', 0.7)
        single_llm = setup_llm(llm_type, model=model, api_key=api_key, temperature=temperature)
        student_llm = single_llm
        teacher_llm = None
        mode = "single"
        print(f"Running single-model benchmark with {model}")

    # === 3️⃣ 初始化 benchmark ===
    benchmark = BooleanBenchmarkRefined(args.dataset)

    # === 4️⃣ 输出路径 ===
    output_pattern = benchmark_cfg.get("output_pattern", "results/{dataset_name}_{model}.json")
    dataset_name = Path(args.dataset).stem
    output_file = args.output or output_pattern.format(
        dataset_name=dataset_name,
        model=mode
    )
    Path(Path(output_file).parent).mkdir(parents=True, exist_ok=True)

    # === 5️⃣ 执行主流程 ===
    if mode == "teacher_student":
        print("\n Running Teacher–Student Boolean benchmark...")
        results = benchmark.run_benchmark(
            llm=student_llm,
            teacher_llm=teacher_llm,
            n_samples=args.n_samples,
            n_queries_per_sample=args.n_queries,
            query_multiplier=args.query_multiplier,
            seed=args.seed,
            verbose=args.verbose,
            checkpoint_dir=args.checkpoint_dir,
            max_retries=3
        )
    else:
        print("\n Running single-model Boolean benchmark...")
        results = benchmark.run_benchmark(
            llm=student_llm,
            n_samples=args.n_samples,
            n_queries_per_sample=args.n_queries,
            query_multiplier=args.query_multiplier,
            seed=args.seed,
            verbose=args.verbose,
            checkpoint_dir=args.checkpoint_dir,
            max_retries=3
        )

    # === 6️⃣ 保存结果 ===
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Benchmark finished! Results saved to: {output_file}")


if __name__ == "__main__":
    main()