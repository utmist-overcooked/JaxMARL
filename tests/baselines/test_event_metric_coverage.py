"""Every event the environment emits must be logged as a sum, not a mean.

The trainers reduce the whole info dict with .mean() and then overwrite only the
names in EVENT_METRIC_NAMES with .sum(). An event missing from that list is
therefore silently divided by NUM_STEPS * NUM_ACTORS in W&B - which is how the
dish-washing events came to look like they never happened.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from baselines.IPPO.ippo_cnn_overcooked_v3 import EVENT_METRIC_NAMES  # noqa: E402
from jaxmarl.environments.overcooked_v3.settings import EVENT_NAMES  # noqa: E402


def test_every_env_event_is_summed_in_logging():
    missing = sorted(set(EVENT_NAMES) - set(EVENT_METRIC_NAMES))
    assert not missing, (
        "these events would be logged as batch means instead of sums: " f"{missing}"
    )
