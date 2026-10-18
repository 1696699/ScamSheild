"""Create fictional, non-deployable SMS examples for classifier evaluation."""

import random
from itertools import product

SCAM_SERVICES = (
    ("Morrowmere Parcel", "morrowmere-parcel.invalid"),
    ("Bellwether Mobile", "bellwether-mobile.invalid"),
    ("Cindercone Rewards", "cindercone-rewards.invalid"),
    ("Larkspur Billing", "larkspur-billing.invalid"),
    ("Pineglass Streaming", "pineglass-streaming.invalid"),
    ("Willowmere Delivery", "willowmere-delivery.invalid"),
    ("North Quay Utilities", "north-quay-utilities.invalid"),
    ("Marble Finch Tickets", "marble-finch-tickets.invalid"),
    ("Westmere Accounts", "westmere-accounts.invalid"),
    ("Daybreak Wireless", "daybreak-wireless.invalid"),
)
SCAM_EVENTS = (
    "a delivery update is waiting",
    "a service notice was added this morning",
    "the monthly reward list has changed",
    "a billing notice is ready to view",
    "your account alert needs a quick review",
    "one parcel detail changed after dispatch",
    "an unusual sign-in notice was recorded",
    "your entry was selected for a sample prize draw",
    "the service status changed overnight",
    "a final reminder is available before today's cutoff",
)
SCAM_ACTIONS = (
    "Details: {url}",
    "Review the notice at {url}",
    "More info here: {url}",
    "The update is here: {url}",
)

LEGITIMATE_SERVICES = (
    ("Mossbank Book Circle", "mossbank-books.invalid"),
    ("Foxglove Neighbourhood Club", "foxglove-neighbours.invalid"),
    ("Copperfield Garden Group", "copperfield-garden.invalid"),
    ("Bluefern Running Team", "bluefern-running.invalid"),
    ("Maplewick Music Society", "maplewick-music.invalid"),
    ("Pebblebrook Clinic", "pebblebrook-clinic.invalid"),
    ("Sundial Community Hall", "sundial-hall.invalid"),
    ("Thistlewood Chess Club", "thistlewood-chess.invalid"),
    ("Birch Lantern Library", "birchlantern-library.invalid"),
    ("Cloudberry Art Group", "cloudberry-art.invalid"),
)
LEGITIMATE_EVENTS = (
    "the monthly meeting starts at 6 pm as usual",
    "practice has moved to the smaller hall tonight",
    "your appointment is confirmed for Tuesday morning",
    "the book discussion is now on Thursday",
    "the picnic has moved to Saturday because of the weather",
    "the library books are ready to collect",
    "the choir will meet ten minutes later this week",
    "the garden group is bringing tea after the talk",
    "the match starts at the usual time on Sunday",
    "the room booking is confirmed for next week",
)
LEGITIMATE_CLOSINGS = (
    "See you there.",
    "No reply needed.",
    "Reply if you need the notes.",
    "The usual details are at {url}.",
)

FRAUD_MESSAGES = tuple(
    f"{service}: {event}. "
    f"{action.format(url=f'https://{domain}/notice')}"
    for (service, domain), event, action in product(
        SCAM_SERVICES, SCAM_EVENTS, SCAM_ACTIONS
    )
)
LEGITIMATE_MESSAGES = tuple(
    f"{service}: {event}. "
    f"{closing.format(url=f'https://{domain}/details')}"
    for (service, domain), event, closing in product(
        LEGITIMATE_SERVICES, LEGITIMATE_EVENTS, LEGITIMATE_CLOSINGS
    )
)


def generate_fraud_messages(count: int, seed: int = 42) -> list[str]:
    """Return a deterministic, unique batch of fictional scam-like messages."""
    return _sample(FRAUD_MESSAGES, count, seed)


def generate_legitimate_messages(count: int, seed: int = 42) -> list[str]:
    """Return a deterministic, unique batch of legitimate lookalikes."""
    return _sample(LEGITIMATE_MESSAGES, count, seed)


def _sample(messages: tuple[str, ...], count: int, seed: int) -> list[str]:
    if count < 1 or count > len(messages):
        raise ValueError(f"Example count must be between 1 and {len(messages)}")
    return random.Random(seed).sample(messages, count)
