import sys
import json
import argparse
import os
import re
import yaml
import random
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any
from datetime import datetime
from textwrap import dedent
from collections import Counter
from scipy import stats
import traceback

from modules.models import CausalGraph
from modules.llm_interface import LLMInterface, OpenRouterLLM, OpenAILLM, AnthropicLLM
from generate_causal_dataset import PerturbationObservation, CausalDatasetGenerator





# Add this above setup_llm()
class DeepSeekLLM:
    """LLM interface for DeepSeek official API."""
    def __init__(self, model, api_key, temperature=0.7):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model
        self.temperature = temperature

    def query(self, prompt: str) -> str:
        """Send a single query to DeepSeek API."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            stream=False
        )
        return response.choices[0].message.content

    def get_name(self):
        return self.model


class CausalBenchmarkEnhanced:
    """Enhanced benchmark for evaluating LLM creativity in causal graph discovery with comprehensive tracking."""
    
    def __init__(self, complete_dataset_path: Optional[str] = None, 
                 n_observations_filter: Optional[List[int]] = None,
                 gt_filter: Optional[Tuple] = None):
        """
        Initialize benchmark with a complete dataset or empty.
        
        Args:
            complete_dataset_path: Path to the complete causal dataset JSON file (optional)
            n_observations_filter: List of n_observations values to include (optional)
            gt_filter: Tuple of (min, max) for GT range or (list, None) for specific values (optional)
        """
        self.n_observations_filter = n_observations_filter
        self.gt_filter = gt_filter
        self.filtered_observation_sets = []
        self.excluded_observation_sets = []  # For potential backfill
        
        if complete_dataset_path:
            with open(complete_dataset_path, 'r') as f:
                self.complete_dataset = json.load(f)
            
            # Extract metadata
            self.metadata = self.complete_dataset.get('metadata', {})
            self.nodes = self.metadata.get('nodes', [])
            self.max_edges = self.metadata.get('max_edges', None)
            
            # Flatten all datasets into a single list for sampling
            self.all_observation_sets = []
            if 'datasets_by_n_observations' in self.complete_dataset:
                for n_obs, datasets in self.complete_dataset['datasets_by_n_observations'].items():
                    self.all_observation_sets.extend(datasets)
            elif 'datasets' in self.complete_dataset:
                self.all_observation_sets = self.complete_dataset['datasets']
            elif 'sampled_datasets' in self.complete_dataset:
                self.all_observation_sets = self.complete_dataset['sampled_datasets']
            
            # Apply two-stage filtering
            stage1_filtered = self.all_observation_sets
            
            # Stage 1: Apply n_observations filter if specified
            if n_observations_filter:
                stage1_filtered = [
                    obs_set for obs_set in self.all_observation_sets
                    if obs_set.get('n_observations') in n_observations_filter
                ]
                print(f"Stage 1: Filtered to {len(stage1_filtered)} observation sets with n_observations in {n_observations_filter}")
            
            # Stage 2: Apply GT filter if specified
            if gt_filter:
                if gt_filter[1] is not None:  # Range filter (min, max)
                    min_gt, max_gt = gt_filter
                    self.filtered_observation_sets = [
                        obs_set for obs_set in stage1_filtered
                        if min_gt <= obs_set.get('n_compatible_graphs', 0) <= max_gt
                    ]
                    print(f"Stage 2: Filtered to {len(self.filtered_observation_sets)} observation sets with n_compatible_graphs in [{min_gt}, {max_gt}]")
                else:  # Specific values filter
                    allowed_values = gt_filter[0] if isinstance(gt_filter[0], list) else []
                    self.filtered_observation_sets = [
                        obs_set for obs_set in stage1_filtered
                        if obs_set.get('n_compatible_graphs', 0) in allowed_values
                    ]
                    print(f"Stage 2: Filtered to {len(self.filtered_observation_sets)} observation sets with n_compatible_graphs in {allowed_values}")
                
                # Keep excluded sets for potential backfill
                self.excluded_observation_sets = [
                    obs_set for obs_set in self.all_observation_sets
                    if obs_set not in self.filtered_observation_sets
                ]
            else:
                self.filtered_observation_sets = stage1_filtered
                self.excluded_observation_sets = []
            
            # Infer max_edges from ground truth graphs if not specified
            if self.max_edges is None and self.all_observation_sets:
                max_edges_in_gts = 0
                for obs_set in self.all_observation_sets:
                    for gt in obs_set.get('ground_truth_graphs', []):
                        num_edges = len(gt.get('edges', []))
                        max_edges_in_gts = max(max_edges_in_gts, num_edges)
                self.max_edges = max_edges_in_gts
                print(f"Inferred max_edges={self.max_edges} from ground truth graphs")
            
            print(f"Loaded complete dataset with {len(self.all_observation_sets)} observation sets")
            if n_observations_filter or gt_filter:
                print(f"After filtering: {len(self.filtered_observation_sets)} observation sets meet criteria")
                if self.excluded_observation_sets:
                    print(f"  ({len(self.excluded_observation_sets)} observation sets available for backfill)")
            print(f"Nodes: {', '.join(self.nodes)}")
            print(f"Max edges in hypothesis space: {self.max_edges if self.max_edges is not None else 'unlimited'}")
        else:
            self.complete_dataset = None
            self.all_observation_sets = []
            self.filtered_observation_sets = []
            print("Initialized empty benchmark - will generate datasets on demand")
    
    def sample_observation_sets(self, n_samples: int, seed: Optional[int] = None) -> List[Dict]:
        """
        Sample n observation sets from the complete dataset with smart backfill.
        
        If filters are applied and not enough datasets meet criteria, backfill from excluded sets.
        
        Args:
            n_samples: Number of observation sets to sample
            seed: Random seed for reproducibility
        
        Returns:
            List of sampled observation sets
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Determine primary pool based on filters
        if self.n_observations_filter or self.gt_filter:
            primary_pool = self.filtered_observation_sets
            backup_pool = self.excluded_observation_sets
        else:
            primary_pool = self.all_observation_sets
            backup_pool = []
        
        sampled = []
        
        # First, sample from primary pool (datasets meeting filter criteria)
        n_primary = len(primary_pool)
        if n_primary > 0:
            n_from_primary = min(n_samples, n_primary)
            sampled_primary = random.sample(primary_pool, n_from_primary)
            sampled.extend(sampled_primary)
            
            # Mark these as meeting filter criteria
            for obs_set in sampled_primary:
                obs_set['meets_filter_criteria'] = True
        
        # If we need more samples, backfill from backup pool
        n_still_needed = n_samples - len(sampled)
        if n_still_needed > 0 and backup_pool:
            n_backup = len(backup_pool)
            n_from_backup = min(n_still_needed, n_backup)
            
            if n_from_backup > 0:
                print(f"\nBackfilling: Only {n_primary} datasets met filter criteria.")
                print(f"  Adding {n_from_backup} randomly selected datasets from outside the filter range.")
                
                sampled_backup = random.sample(backup_pool, n_from_backup)
                
                # Mark these as backfilled
                for obs_set in sampled_backup:
                    obs_set['meets_filter_criteria'] = False
                    obs_set['backfilled'] = True
                
                sampled.extend(sampled_backup)
        
        # Final check if we still don't have enough
        if len(sampled) < n_samples:
            total_available = len(primary_pool) + len(backup_pool)
            print(f"\nWarning: Requested {n_samples} samples but only {total_available} total datasets available.")
            print(f"  Returning {len(sampled)} datasets.")
        
        return sampled
    
    def create_prompt(self, observations: List[Dict], prior_hypotheses: List[CausalGraph],enable_cot: bool = True) -> str:
        """Create prompt for LLM."""
        nodes_str = ", ".join(self.nodes)
        obs_block = "\n".join(obs["string"] for obs in observations)
        
        if prior_hypotheses:
            prior_lines = []
            for h in prior_hypotheses:
                edges = [f"{s}->{d}" for s, d in h.edges]
                if edges:
                    prior_lines.append("Graph: " + ", ".join(edges))
                else:
                    prior_lines.append("Graph: No edges")
            prior_block = "\n".join(prior_lines)
        else:
            prior_block = "None"
        
        # Add constraint information if max_edges is known
        constraint_info = ""
        if self.max_edges is not None:
            constraint_info = f"\nConstraint: The graph should have at most {self.max_edges} edges."
            # Added: Implicit chain-of-thought + self-check repair (output only final line)
        cot_block = ""
        if enable_cot:
            cot_block = """
            Internal Deliberation (do not reveal):
            - For each perturbation P, compute the descendant set Desc(P) implied by your graph.
            - Ensure nodes marked 1 in observations equal Desc(P) exactly; nodes marked 0 are not descendants.
            - If any mismatch exists, adjust the fewest edges possible (no cycles, obey max edge constraint) until all perturbations match.
            - Do NOT output your reasoning or steps. Only output the final graph line.
            """.strip()
        
        prompt = f"""
        You are given observations from perturbation experiments on a causal system.
        
        Semantics:
        - When a node is perturbed, the perturbed node is 0.
        - A node is 1 if it is a downstream descendant of the perturbed node in the causal graph.
        - All other nodes are 0.
        
        Nodes: {nodes_str}{constraint_info}
        
        Observations:
        {obs_block}
        
        Prior predictions (do not repeat if avoidable):
        {prior_block}


        Task:
        Output a single directed acyclic graph (DAG) over the nodes above that explains all observations.
        
        Diversity rule:
        - A "diverse" graph is any valid graph whose edge set is NOT identical to any prior prediction.
        - Generate diverse graphs when possible to explore the solution space.
        ############# 
        
        Formatting rules:
        1) Use only the listed nodes. No self-loops. No cycles.
        2) Respond with exactly one line:
        - If there are edges: Graph: A->B, B->C
        - If there are no edges: Graph: No edges
        """
        return dedent(prompt).strip()
    
    def parse_llm_response(self, response: str) -> Optional[CausalGraph]:
        """Parse LLM response to extract causal graph."""
        if not isinstance(response, str):
            return None
        
        # Strip code fences and whitespace
        s = response.replace("```", "").strip()
        
        # Look for "Graph:" line
        m = re.search(r'(?i)\bgraph\s*:\s*(.+)$', s, flags=re.MULTILINE)
        line = m.group(1).strip() if m else (s.splitlines()[0].strip() if s.splitlines() else "")
        if not line:
            return None
        
        # Clean up the line
        if (line.startswith('"') and line.endswith('"')) or (line.startswith("'") and line.endswith("'")):
            line = line[1:-1].strip()
        line = (line
                .replace("→", "->")
                .replace("-->", "->")
                .replace("=>", "->")
                .rstrip(" .;"))
        
        # Handle "No edges"
        if re.search(r'\b(no\s+edges?|empty|none|null)\b', line, re.I):
            return CausalGraph(nodes=self.nodes, edges=[])
        
        # Parse edges
        parts = [p.strip() for p in line.split(",") if p.strip()]
        if not parts:
            return None
        
        edges = []
        for part in parts:
            m = re.fullmatch(r'([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)', part)
            if not m:
                return None
            u, v = m.group(1), m.group(2)
            if u not in self.nodes or v not in self.nodes or u == v:
                return None
            edges.append((u, v))
        
        # Deduplicate
        edges = list(dict.fromkeys(edges))
        
        # Check max_edges constraint if specified
        if self.max_edges is not None and len(edges) > self.max_edges:
            return None  # Too many edges
        
        # Validate DAG
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(edges)
        if not nx.is_directed_acyclic_graph(G):
            return None
        
        return CausalGraph(nodes=self.nodes, edges=edges)
    
    def validate_hypothesis(self, hypothesis: CausalGraph, observations: List[Dict]) -> bool:
        """Check if hypothesis is consistent with observations."""
        for obs_dict in observations:
            perturbed_node = obs_dict['perturbed_node']
            expected_effects = obs_dict['effects']
            
            # Get what this hypothesis would produce
            hypothesis_obs = CausalDatasetGenerator.get_perturbation_effects(hypothesis, perturbed_node)
            
            # Check if effects match
            if hypothesis_obs.effects != expected_effects:
                return False
        
        return True
    
    def _classify_error(self, error_message: str) -> str:
        """Classify the type of error from the error message."""
        if "Expecting value" in error_message:
            match = re.search(r'line (\d+) column (\d+)', error_message)
            if match:
                return f"json_parse_error (line {match.group(1)}, col {match.group(2)})"
            return "json_parse_error"
        elif "Rate limit" in error_message.lower() or "rate_limit" in error_message.lower():
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
            match = re.search(r'\b(\d{3})\b', error_message)
            if match:
                return f"http_error_{match.group(1)}"
            return "unknown_error"
    
    def evaluate_single_observation_set(
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
        # 兼容两种参数名
        if observation_set is not None:
            observations = observation_set
        elif observations is not None:
            observation_set = observations
        else:
            raise ValueError("Either observation_set or observations must be provided")
        """
        Evaluate LLMs on a single observation set with a Teacher–Student refinement loop.
        Student generates -> Teacher critiques -> Student refines.
        """

        # === 基础数据提取 ===
        observations = observation_set['observations']
        ground_truth_graphs = [
            CausalGraph.from_dict(g) for g in observation_set['ground_truth_graphs']
        ]
        gt_hashes = {g.get_hash() for g in ground_truth_graphs}

        # === 结果追踪 ===
        all_hypotheses = []
        valid_hypotheses = []
        unique_hashes = set()
        unique_valid_graphs = []
        all_unique_hashes = set()
        unique_all_graphs = []
        parse_success_count = 0
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
            print(f"\nQuery {i+1}/{n_queries}")
            print(f"  Step 1: Student generating initial hypothesis...")

            # 确保正确提取观察数据
            obs_data = observation_set['observations'] if isinstance(observation_set, dict) and 'observations' in observation_set else observation_set
            # 提供空的prior_hypotheses列表
            prompt = self.create_prompt(observations=obs_data, prior_hypotheses=[])
            hypothesis = None
            query_error = None
            
            try:
                student_result = student_llm.query(prompt)
                print(f"  Step 2: Teacher reviewing hypothesis...")
            except Exception as e:
                query_error = {'query_index': i, 'error_message': str(e)}
                errors.append(query_error)
                print(f"  Error: Failed to get student response: {str(e)}")
                continue

            # --- Step 2: 老师审阅反馈 ---
            feedback_prompt = f"""
            You are an expert causal reasoning teacher reviewing a student's proposed DAG.

            IMPORTANT SEMANTICS OF PERTURBATION EXPERIMENTS:
            - When a node is perturbed, the perturbed node is 0
            - A node is 1 if and only if it is a downstream descendant of the perturbed node in the causal graph
            - All other nodes are 0

            Observations (perturbation data):
            {json.dumps(obs_data, indent=2)}

            Student's proposed graph:
            ```
            {student_result}
            ```

            Please analyze whether this graph explains the observations logically.
            IMPORTANT RULES FOR CAUSAL ANALYSIS:
            1. If node X causes Y (X->Y), then perturbing X should make Y=1 (since Y is downstream of X)
            2. If node X causes Y (X->Y), perturbing Y should make X=0 (since X is NOT downstream of Y)
            3. If perturbing X results in Y=1, then Y must be downstream of X in the graph
            4. If perturbing X results in Y=0, then Y is either the perturbed node itself or not downstream of X
            5. Carefully analyze all perturbation experiments to validate edges
            6. Only suggest edits when there is clear evidence of inconsistency
            7. If the student's graph already correctly explains all observations, confirm it's correct
            
            Example reasoning:
            - If perturbing B gives B=0, A=1, C=0, this means A must be downstream of B (B->A)
            - The reverse (A->B) would mean perturbing A would make B=1, and perturbing B would make A=0
            
            - Identify any inconsistencies, missing links, or unnecessary edges.
            - Suggest concrete edits (add/remove/reverse edges) to make it valid and parsimonious.
            - Keep your feedback short and actionable.
            """
            try:
                teacher_feedback = teacher_llm.query(feedback_prompt)
                print(f"  Step 3: Student refining hypothesis...")
            except Exception as e:
                teacher_feedback = f"(Teacher failed: {str(e)})"
                print(f"  Warning: Teacher feedback generation failed: {str(e)}")

            # --- Step 3: 学生根据反馈修订 ---
            refine_prompt = f"""
            You previously proposed this graph:
            {student_result}

            Your teacher provided this feedback:
            "{teacher_feedback}"

            Now produce ONE revised causal graph that addresses the feedback
            and fits all observations. Keep it acyclic and minimal.

            Respond with exactly one line:
            - If there are edges: Graph: A->B, B->C
            - If there are no edges: Graph: No edges
            """
            try:
                refined_output = student_llm.query(refine_prompt)
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
            hypothesis = self.parse_llm_response(refined_output)
            if hypothesis:
                parse_success_count += 1
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
            all_hypotheses.append(hypothesis)

            # 检查 uniqueness（所有）
            all_h_hash = hypothesis.get_hash()
            if all_h_hash not in all_unique_hashes:
                all_unique_hashes.add(all_h_hash)
                unique_all_graphs.append(hypothesis)

            # 验证有效性
            is_valid = self.validate_hypothesis(hypothesis, observations)
            if is_valid:
                valid_hypotheses.append(hypothesis)
                h_hash = hypothesis.get_hash()
                if h_hash not in unique_hashes:
                    unique_hashes.add(h_hash)
                    unique_valid_graphs.append(hypothesis)
                print(f"Hypothesis is valid")
            else:
                print(f"Hypothesis is invalid")

        # === 统计指标 ===
        valid_rate = len(valid_hypotheses) / n_queries if n_queries > 0 else 0
        novelty_rate = len(unique_all_graphs) / n_queries if n_queries > 0 else 0
        parse_success_rate = parse_success_count / n_queries if n_queries > 0 else 0

        # === Recovery 计算 ===
        recovered_gts = set()
        for graph in unique_valid_graphs:
            if graph.get_hash() in gt_hashes:
                recovered_gts.add(graph.get_hash())

        recovery_rate = len(recovered_gts) / len(gt_hashes) if gt_hashes else 0

        # === 输出结构 ===
        return {
            'teacher_feedbacks': teacher_feedbacks,
            'observation_set_id': observation_set.get('observation_set_id', 'unknown'),
            'n_observations': len(observations),
            'n_ground_truths': len(ground_truth_graphs),
            'n_queries': n_queries,
            'n_valid': len(valid_hypotheses),
            'n_unique_valid': len(unique_valid_graphs),
            'n_unique_all': len(unique_all_graphs),
            'n_recovered_gts': len(recovered_gts),
            'parse_success_count': parse_success_count,
            'parse_success_rate': parse_success_rate,
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
            'all_hypotheses': [h.to_dict() for h in all_hypotheses],
            'valid_hypotheses': [h.to_dict() for h in valid_hypotheses],
            'unique_graphs': [g.to_dict() for g in unique_valid_graphs]
        }

    def run_benchmark(
        self,
        llm: LLMInterface,
        n_samples: int = 10,
        n_queries_per_sample: Optional[int] = None,
        query_multiplier: float = 2.0,
        seed: Optional[int] = None,
        verbose: bool = True,
        checkpoint_dir: str = "checkpoints",
        max_retries: int = 3
    ) -> Dict:
        """
        Run the enhanced benchmark with comprehensive tracking.
        
        Args:
            llm: LLM interface to use
            n_samples: Number of observation sets to sample
            n_queries_per_sample: Fixed number of queries per observation set
            query_multiplier: Multiplier for n_gt to determine queries (default 2.0)
            seed: Random seed
            verbose: Print progress
            checkpoint_dir: Directory to save checkpoints
            max_retries: Maximum retries per query
        
        Returns:
            Dictionary with complete benchmark results including statistics, token usage, and costs
        """
        # Create checkpoint directory
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Generate run ID and checkpoint file
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_llm_name = llm.get_name().replace('/', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        checkpoint_file = checkpoint_path / f"checkpoint_causal_enhanced_{safe_llm_name}_{run_id}.json"
        
        print(f"\nRunning Enhanced Causal Benchmark")
        print(f"LLM: {llm.get_name()}")
        print(f"Sampling {n_samples} observation sets")
        if n_queries_per_sample is not None:
            print(f"Queries per sample: {n_queries_per_sample} (fixed)")
        else:
            print(f"Queries per sample: {query_multiplier}x number of ground truths (adaptive)")
        print(f"Max retries: {max_retries}")
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
        
        # Token and cost tracking
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        
        # Error tracking
        all_errors = []
        total_error_counts = {}
        
        # Load existing checkpoint if it exists
        start_idx = 0
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    all_results = checkpoint_data.get('results', [])
                    start_idx = len(all_results)
                    
                    # Restore token/cost data
                    total_prompt_tokens = checkpoint_data.get('total_prompt_tokens', 0)
                    total_completion_tokens = checkpoint_data.get('total_completion_tokens', 0)
                    total_tokens = checkpoint_data.get('total_tokens', 0)
                    total_cost = checkpoint_data.get('total_cost', 0.0)
                    
                    # Restore error data
                    all_errors = checkpoint_data.get('all_errors', [])
                    total_error_counts = checkpoint_data.get('total_error_counts', {})
                    
                    print(f"Resuming from checkpoint: {start_idx}/{n_samples} completed")
                    
                    # Recalculate rates from checkpoint
                    for result in all_results:
                        valid_rates.append(result['valid_rate'])
                        novelty_rates.append(result['novelty_rate'])
                        recovery_rates.append(result['recovery_rate'])
                        parse_success_rates.append(result.get('parse_success_rate', 1.0))
            except Exception as e:
                print(f"Warning: Failed to load checkpoint: {e}")
                print("Starting from beginning...")
        
        # Process each sampled observation set
        for idx in range(start_idx, len(sampled_sets)):
            obs_set = sampled_sets[idx]
            
            if verbose:
                print(f"\nSample {idx + 1}/{n_samples}")
                print(f"  Observation set ID: {obs_set.get('observation_set_id', 'unknown')}")
                print(f"  Number of observations: {len(obs_set['observations'])}")
                print(f"  Number of ground truths: {obs_set['n_compatible_graphs']}")
            
            try:
                # Determine number of queries
                if n_queries_per_sample is not None:
                    n_queries = n_queries_per_sample
                else:
                    n_gt = obs_set['n_compatible_graphs']
                    n_queries = max(1, int(n_gt * query_multiplier))
                    if verbose:
                        print(f"  Using {n_queries} queries ({query_multiplier}x {n_gt} ground truths)")
                
                # Evaluate
                result = self.evaluate_single_observation_set(
                student_llm=llm, 
                teacher_llm=llm,  # Using same LLM as teacher and student
                observation_set=obs_set, 
                n_queries=n_queries, 
                verbose=False, 
                max_retries=max_retries
            )
                
                all_results.append(result)
                valid_rates.append(result['valid_rate'])
                novelty_rates.append(result['novelty_rate'])
                recovery_rates.append(result['recovery_rate'])
                parse_success_rates.append(result['parse_success_rate'])
                
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
                    print(f"  Parse success rate: {result['parse_success_rate']:.2%}")
                    print(f"  Valid rate: {result['valid_rate']:.2%}")
                    print(f"  Novelty rate: {result['novelty_rate']:.2%}")
                    print(f"  Recovery rate: {result['recovery_rate']:.2%}")
                    if result['cost'] > 0:
                        print(f"  Cost: ${result['cost']:.6f}")
                
                # Save checkpoint
                checkpoint_data = {
                    'run_id': run_id,
                    'llm_name': llm.get_name(),
                    'n_samples': n_samples,
                    'n_queries_per_sample': n_queries_per_sample,
                    'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
                    'seed': seed,
                    'timestamp': datetime.now().isoformat(),
                    'results': all_results,
                    'total_prompt_tokens': total_prompt_tokens,
                    'total_completion_tokens': total_completion_tokens,
                    'total_tokens': total_tokens,
                    'total_cost': total_cost,
                    'all_errors': all_errors,
                    'total_error_counts': total_error_counts
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
            'n_samples': len(all_results),
            'n_queries_per_sample': n_queries_per_sample,
            'query_multiplier': query_multiplier if n_queries_per_sample is None else None,
            'query_mode': 'fixed' if n_queries_per_sample is not None else f'adaptive_{query_multiplier}x',
            'seed': seed,
            'timestamp': datetime.now().isoformat(),
            'max_edges_constraint': self.max_edges,
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
        
        # Print comprehensive summary
        print("\n" + "=" * 60)
        print("ENHANCED BENCHMARK RESULTS SUMMARY")
        print("=" * 60)
        print(f"Samples evaluated: {len(all_results)}/{n_samples}")
        print(f"Max edges constraint: {self.max_edges if self.max_edges is not None else 'unlimited'}")
        
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
        
        print(f"\nToken Usage:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Prompt tokens: {total_prompt_tokens:,}")
        print(f"  Completion tokens: {total_completion_tokens:,}")
        print(f"  Avg tokens/sample: {final_results['token_usage']['avg_tokens_per_sample']:.1f}")
        print(f"  Avg tokens/query: {final_results['token_usage']['avg_tokens_per_query']:.1f}")
        
        print(f"\nCost:")
        print(f"  Total cost: ${total_cost:.4f}")
        print(f"  Avg cost/sample: ${final_results['cost']['avg_cost_per_sample']:.4f}")
        print(f"  Avg cost/query: ${final_results['cost']['avg_cost_per_query']:.6f}")
        
        if all_errors:
            print(f"\nErrors:")
            print(f"  Total errors: {len(all_errors)}")
            print(f"  Error rate: {final_results['error_summary']['error_rate']:.2%}")
            if total_error_counts:
                print(f"  Error types:")
                for error_type, count in sorted(total_error_counts.items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {error_type}: {count}")
        
        print("=" * 60)
        
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
            model=kwargs.get('model', 'gpt-4o'),
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
        # Added DeepSeek official API support
    elif llm_type == "deepseek":
        api_key = kwargs.get('api_key') or os.environ.get('DEEPSEEK_API_KEY')
        if not api_key:
            raise ValueError("DeepSeek API key required")

        return DeepSeekLLM(
            model=kwargs.get('model', 'deepseek-chat'),
            api_key=api_key,
            temperature=kwargs.get('temperature', 0.7)
        )

    else:
        raise ValueError(f"Unknown LLM type: {llm_type}")


def parse_n_observations_filter(filter_str: str) -> List[int]:
    """
    Parse n_observations filter string.
    
    Args:
        filter_str: String like "2,3,5" or "2-5" or "2,4-6,8"
        
    Returns:
        List of n_observations values to include
    """
    if not filter_str:
        return []
    
    result = []
    parts = filter_str.split(',')
    
    for part in parts:
        part = part.strip()
        if '-' in part and not part.startswith('-'):
            # Range like "2-5"
            start, end = part.split('-')
            start, end = int(start.strip()), int(end.strip())
            result.extend(range(start, end + 1))
        else:
            # Single value
            result.append(int(part))
    
    # Remove duplicates and sort
    return sorted(list(set(result)))


def parse_gt_filter(filter_str: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse ground truth filter string.
    
    Args:
        filter_str: String like "10-16" for range or "1,2,4" for specific values
        
    Returns:
        Tuple of (min_gt, max_gt) for range, or (values_list, None) for specific values
    """
    if not filter_str:
        return None, None
    
    # Check if it's a range (single dash not at start)
    if '-' in filter_str and not filter_str.startswith('-'):
        parts = filter_str.split('-')
        if len(parts) == 2:
            try:
                min_gt = int(parts[0].strip())
                max_gt = int(parts[1].strip())
                return min_gt, max_gt
            except ValueError:
                pass
    
    # Otherwise, treat as comma-separated list
    try:
        values = []
        for part in filter_str.split(','):
            values.append(int(part.strip()))
        return sorted(values), None
    except ValueError:
        print(f"Warning: Invalid GT filter format: {filter_str}")
        return None, None


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Run causal discovery benchmark with Teacher–Student LLM pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--dataset", required=True, help="Path to causal dataset JSON file")
    parser.add_argument("--config", required=True, help="Path to YAML configuration file")
    parser.add_argument("--n-samples", type=int, default=10, help="Number of observation sets to sample")
    parser.add_argument("--n-queries", type=int, default=None, help="Number of Student–Teacher refinement rounds per sample (optional, will use dynamic calculation if not specified)")
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
    benchmark = CausalBenchmarkEnhanced(args.dataset)

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
        print("\n Running Teacher–Student causal benchmark...")
        sampled_sets = benchmark.sample_observation_sets(args.n_samples, args.seed)
        all_results = []

        for i, obs_set in enumerate(sampled_sets):
            print(f"\n===== Sample {i+1}/{args.n_samples} =====")
            print(f"  Observation set ID: {obs_set.get('observation_set_id', 'unknown')}")
            print(f"  Number of observations: {len(obs_set['observations'])}")
            print(f"  Number of ground truths: {obs_set['n_compatible_graphs']}")
            
            # 根据ground truth数量动态确定查询次数
            n_ground_truths = obs_set['n_compatible_graphs']
            # 使用命令行指定的查询次数（如果提供），否则使用动态计算
            if args.n_queries is not None:
                dynamic_n_queries = args.n_queries
                print(f"  Running {dynamic_n_queries} queries (user specified)...")
            else:
                # 基础公式：查询次数 = ground truth数量 * 1 + 1，最小为1次
                dynamic_n_queries = max(1, n_ground_truths * 1 )
                # 已在上面处理打印逻辑
            print(f"  Running {dynamic_n_queries} queries (dynamically determined based on {n_ground_truths} ground truths)...")
            
            result = benchmark.evaluate_single_observation_set(
                student_llm=student_llm,
                teacher_llm=teacher_llm,
                observation_set=obs_set,
                n_queries=dynamic_n_queries,
                verbose=args.verbose
            )
            
            # 确保在新的一行开始打印统计信息
            print("\n  ===== Results =====")
            print(f"  Parse success rate: {result['parse_success_rate']*100:.2f}%")
            print(f"  Valid rate: {result['valid_rate']*100:.2f}%")
            print(f"  Novelty rate: {result['novelty_rate']*100:.2f}%")
            print(f"  Recovery rate: {result['recovery_rate']*100:.2f}%")
            print("  ===================")
            
            all_results.append(result)

        # 汇总结果
        # 计算统计指标
        statistics = {}
        metrics = ['parse_success_rate', 'valid_rate', 'novelty_rate', 'recovery_rate']
        
        for metric in metrics:
            values = [r[metric] for r in all_results]
            if values:
                # 计算基本统计量
                mean_val = np.mean(values)
                std_val = np.std(values)
                var_val = np.var(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                # 计算p值 (假设检验: 值是否大于0)
                if len(values) > 1:
                    # 使用t检验
                    _, p_value = stats.ttest_1samp(values, 0, alternative='greater')
                else:
                    p_value = 0.5 if mean_val > 0 else 1.0
                
                statistics[metric] = {
                    "mean": float(mean_val),
                    "std": float(std_val),
                    "var": float(var_val),
                    "min": float(min_val),
                    "max": float(max_val),
                    "p_value": float(p_value)
                }
            else:
                statistics[metric] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "var": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "p_value": 1.0
                }
        
        # 记录每个样本实际使用的查询次数
        actual_n_queries_used = [len(r.get('hypotheses', [])) for r in all_results]
        
        results = {
            "mode": "teacher_student",
            "n_samples": args.n_samples,
            "specified_n_queries": args.n_queries,  # 记录用户指定的查询次数（如果有）
            "actual_n_queries_used": actual_n_queries_used,  # 记录每个样本实际使用的查询次数
            "results": all_results,
            "statistics": statistics
        }
        
        # 打印统计摘要
        print("\n===== Overall Statistics =====")
        for metric in metrics:
            stats_data = statistics[metric]
            print(f"{metric}:")
            print(f"  Mean: {stats_data['mean']:.6f}")
            print(f"  Std: {stats_data['std']:.6f}")
            print(f"  Var: {stats_data['var']:.6f}")
            print(f"  Min: {stats_data['min']:.6f}")
            print(f"  Max: {stats_data['max']:.6f}")
            print(f"  p_value: {stats_data['p_value']:.6e}")
        print("===============================")

    else:
        print("\n Running single-model causal benchmark...")
        results = benchmark.run_benchmark(
            llm=student_llm,
            n_samples=args.n_samples,
            n_queries_per_sample=args.n_queries,
            query_multiplier=1.0,
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