"""Create fictional, non-deployable SMS examples for classifier evaluation."""

import random
from itertools import product

FRAUD_OPENERS = (
    "[FICTIONAL SERVICE] account notice:",
    "[FICTIONAL COURIER] delivery update:",
    "[FICTIONAL REWARDS CLUB] message:",
    "[FICTIONAL BILLING TEAM] alert:",
    "[FICTIONAL STREAMING APP] reminder:",
)
FRAUD_EVENTS = (
    "a package notice is waiting",
    "an account setting needs review",
    "a sample reward is available",
    "an invoice detail needs attention",
    "an unfamiliar sign-in was flagged",
)
FRAUD_ACTIONS = (
    "Review the fictional notice at [SAFE TEST LINK].",
    "Check the sample details at [SAFE TEST LINK].",
    "Open the placeholder account page at [SAFE TEST LINK].",
    "Look over the mock alert at [SAFE TEST LINK].",
)

LEGITIMATE_OPENERS = (
    "[FICTIONAL COMMUNITY GROUP] reminder:",
    "[FICTIONAL BOOK CLUB] update:",
    "[FICTIONAL SPORTS TEAM] note:",
    "[FICTIONAL CLINIC] appointment reminder:",
    "[FICTIONAL FRIEND] message:",
)
LEGITIMATE_EVENTS = (
    "the monthly planning meeting starts at 6 pm",
    "the book-club selection is ready for discussion",
    "practice will begin at the usual time",
    "the picnic has moved to Saturday",
    "your appointment is confirmed for Tuesday at 10 am",
)
LEGITIMATE_CLOSINGS = (
    "See you there.",
    "Reply if you need the notes.",
    "Thanks, see you soon.",
    "No reply needed.",
)

FRAUD_MESSAGES = tuple(
    f"SIMULATED - DO NOT SEND: {opener} {event}. {action}"
    for opener, event, action in product(FRAUD_OPENERS, FRAUD_EVENTS, FRAUD_ACTIONS)
)
LEGITIMATE_MESSAGES = tuple(
    f"SIMULATED - DO NOT SEND: {opener} {event}. {closing}"
    for opener, event, closing in product(
        LEGITIMATE_OPENERS, LEGITIMATE_EVENTS, LEGITIMATE_CLOSINGS
    )
)


def generate_fraud_messages(count: int, seed: int = 42) -> list[str]:
    """Return a deterministic, unique batch of simulated scam-like messages."""
    return _sample(FRAUD_MESSAGES, count, seed)


def generate_legitimate_messages(count: int, seed: int = 42) -> list[str]:
    """Return a deterministic, unique batch of legitimate lookalikes."""
    return _sample(LEGITIMATE_MESSAGES, count, seed)


def _sample(messages: tuple[str, ...], count: int, seed: int) -> list[str]:
    if count < 1 or count > len(messages):
        raise ValueError(f"Example count must be between 1 and {len(messages)}")
    return random.Random(seed).sample(messages, count)