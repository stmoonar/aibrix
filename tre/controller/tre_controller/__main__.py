from __future__ import annotations

import asyncio

from tre_common.logging import setup_json_logging
from tre_controller.app import main


if __name__ == "__main__":
    setup_json_logging()
    asyncio.run(main())
