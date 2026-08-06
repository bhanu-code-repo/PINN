"""Orchestrator — 3-mode runner for Lang-PINN.

Modes:
- **library**: Rule-based pipeline. No LLM calls. PDE Agent parses via LLM,
  but PINN Agent and Code Agent use deterministic rules/templates.
- **code-agent**: Full LLM pipeline. All agents consult the LLM.
- **hybrid**: LLM generates code targeting the ``pinn`` library API,
  with training feedback driving iterative refinement.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field

from llm_provider import LLMClient
from loguru import logger

from .agents.code_agent import CodeAgent
from .agents.pde_agent import PDEAgent
from .agents.pinn_agent import PINNAgent
from .schemas import ArchitectureRec, PDESpec


@dataclass
class SolveResult:
    """Result of an orchestrated solve."""

    spec: PDESpec
    architecture: ArchitectureRec
    code: str
    mode: str
    executed: bool = False
    quality_score: float | None = None
    health_report: dict | None = None
    error: str | None = None
    iterations: int = 1
    history: list[dict] = field(default_factory=list)


class Orchestrator:
    """3-mode Lang-PINN orchestrator.

    Args:
        client: Shared LLM client for all agents.
        mode: Operating mode — ``"library"``, ``"code-agent"``, or ``"hybrid"``.

    Usage::

        orch = Orchestrator(mode="hybrid")
        result = orch.solve("Solve u'' + 2u' + 6400u = 0, u(0)=1, u'(0)=0")
        print(result.code)         # generated Python code
        print(result.quality_score) # if executed
    """

    VALID_MODES = ("library", "code-agent", "hybrid")

    def __init__(
        self,
        client: LLMClient | None = None,
        mode: str = "hybrid",
    ):
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Choose from {self.VALID_MODES}")

        self.mode = mode
        self.client = client or LLMClient()
        self.pde_agent = PDEAgent(self.client)
        self.pinn_agent = PINNAgent(self.client)
        self.code_agent = CodeAgent(self.client)

    def solve(
        self,
        description: str,
        *,
        execute: bool = False,
        max_iterations: int = 3,
        spec: PDESpec | None = None,
    ) -> SolveResult:
        """Run the full Lang-PINN pipeline.

        Args:
            description: Natural language problem description.
            execute: If True, execute the generated code.
            max_iterations: Max refinement iterations (hybrid mode only).
            spec: Pre-parsed PDESpec (skips PDE Agent if provided).

        Returns:
            SolveResult with spec, architecture, code, and optional execution results.
        """
        logger.info("Orchestrator: mode={}, execute={}", self.mode, execute)

        # Step 1: Parse PDE
        if spec is None:
            spec = self.pde_agent.parse(description)
        else:
            logger.info("Orchestrator: using pre-parsed PDESpec '{}'", spec.name)

        # Step 2: Recommend architecture
        use_llm = self.mode in ("code-agent", "hybrid")
        arch = self.pinn_agent.recommend(spec, use_llm=use_llm)
        logger.info("Architecture: {}x{} {}, {} epochs, ansatz={}",
                     arch.hidden_layers, arch.hidden_neurons, arch.activation,
                     arch.epochs, arch.use_ansatz)

        # Step 3: Generate code
        use_llm_code = self.mode in ("code-agent", "hybrid")
        code = self.code_agent.generate(spec, arch, use_llm=use_llm_code)

        result = SolveResult(
            spec=spec,
            architecture=arch,
            code=code,
            mode=self.mode,
        )

        # Step 4: Execute (if requested)
        if execute:
            if self.mode == "hybrid":
                self._execute_with_refinement(result, max_iterations)
            else:
                self._execute_once(result)

        return result

    def solve_from_spec(
        self,
        spec: PDESpec,
        *,
        execute: bool = False,
        max_iterations: int = 3,
    ) -> SolveResult:
        """Solve from a pre-built PDESpec (skips PDE Agent)."""
        return self.solve("", spec=spec, execute=execute,
                          max_iterations=max_iterations)

    def _execute_once(self, result: SolveResult) -> None:
        """Execute the generated code once."""
        logger.info("Executing generated code...")
        try:
            namespace = {}
            exec(result.code, namespace)  # noqa: S102
            result.executed = True

            # Extract quality info if available
            if "quality" in namespace:
                result.quality_score = namespace["quality"].get("quality_score")
            if "monitor" in namespace:
                result.health_report = namespace["monitor"].report()

            logger.info("Execution completed successfully")
        except Exception as e:
            result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            logger.error("Execution failed: {}", result.error)

    def _execute_with_refinement(self, result: SolveResult, max_iterations: int) -> None:
        """Execute with feedback-driven refinement (hybrid mode).

        After each execution, check quality. If below threshold, ask the LLM
        to refine the code based on the error or quality report.
        """
        quality_threshold = 0.5

        for iteration in range(1, max_iterations + 1):
            logger.info("Hybrid iteration {}/{}", iteration, max_iterations)
            result.iterations = iteration

            self._execute_once(result)

            iteration_record = {
                "iteration": iteration,
                "quality_score": result.quality_score,
                "error": result.error,
            }
            result.history.append(iteration_record)

            # Success check
            if result.executed and result.error is None:
                if result.quality_score is not None and result.quality_score >= quality_threshold:
                    logger.info("Quality {:.3f} >= {:.3f} — accepting result",
                                result.quality_score, quality_threshold)
                    return
                elif result.quality_score is None:
                    # No quality score available, accept if no error
                    logger.info("Execution succeeded (no quality score available)")
                    return

            # Refinement: ask LLM to fix the code
            if iteration < max_iterations:
                logger.info("Quality {:.3f} below threshold, requesting refinement...",
                            result.quality_score or 0.0)
                result.code = self._refine_code(result)
                result.executed = False
                result.error = None

        logger.warning("Max iterations reached without meeting quality threshold")

    def _refine_code(self, result: SolveResult) -> str:
        """Ask the LLM to refine code based on execution feedback."""
        feedback_parts = [f"The previous code for '{result.spec.name}' needs improvement."]

        if result.error:
            feedback_parts.append(f"Error: {result.error}")
        if result.quality_score is not None:
            feedback_parts.append(f"Quality score: {result.quality_score:.3f} (target: >= 0.5)")
        if result.health_report:
            feedback_parts.append(f"Health report: {result.health_report}")

        feedback_parts.append(f"\nCurrent code:\n```python\n{result.code}\n```")
        feedback_parts.append(
            "\nPlease fix the issues and return the complete corrected Python code. "
            "Use the same pinn library imports and patterns."
        )

        prompt = "\n".join(feedback_parts)

        from .agents.code_agent import _extract_code
        raw = self.client.ask(prompt, system=(
            "You are a PINN debugging expert. Fix the code based on the feedback. "
            "Return ONLY the corrected Python code, no explanation."
        ))
        return _extract_code(raw)
