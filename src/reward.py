"""Module 4: Fuzzy Proxy Reward for RL training.

Reward = (0.1 * Parses) + (0.3 * TypeChecks) + (0.6 * PassesTests)
"""

ALPHA_PARSES = 0.1
BETA_TYPECHECKS = 0.3
GAMMA_TESTS = 0.6


def compute_reward(parses: bool, type_checks: bool, passes_tests: bool) -> float:
    """Compute the Fuzzy Proxy reward signal.

    Args:
        parses: Whether the generated code parses successfully.
        type_checks: Whether the generated code passes type checking.
        passes_tests: Whether the generated code passes the test suite.

    Returns:
        Scalar reward in [0.0, 1.0].
    """
    return (
        ALPHA_PARSES * float(parses)
        + BETA_TYPECHECKS * float(type_checks)
        + GAMMA_TESTS * float(passes_tests)
    )
