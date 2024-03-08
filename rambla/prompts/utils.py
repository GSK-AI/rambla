import logging
import string
from typing import List

logger = logging.getLogger(__file__)


def find_field_placeholders(text: str) -> List[str]:
    """Finds field placeholders in unformatted strings.

    NOTE: double brackets don't work.
    """
    placeholders = [item[1] for item in string.Formatter().parse(text) if item[1]]
    return placeholders
