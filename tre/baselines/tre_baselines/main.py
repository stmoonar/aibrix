"""Entry point: ``python3 -m tre_baselines.main``."""
from __future__ import annotations

import logging
import signal
import threading

import redis as redis_lib

from tre_baselines.config import load_config
from tre_baselines.loop import BaselineShell, DecisionLog, OwnerLock, make_http_server
from tre_baselines.policies import build_policy
from tre_baselines.sm_client import ACTOR, Dispatcher, SMClient
from tre_baselines.sources import K8sPodLister, LiveSource

LOG = logging.getLogger("tre_baselines")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    config = load_config()
    redis = redis_lib.Redis.from_url(config.redis_url, decode_responses=True, socket_timeout=5.0)
    sm = SMClient(config.sm_url, timeout_s=config.sm_timeout_s, state_timeout_s=config.sm_state_timeout_s,
                  actor=f"{ACTOR}/{config.policy}")
    pods = K8sPodLister(config.model_namespace, port_override=config.metrics_port)
    source = LiveSource(config, redis, sm.get_state, pods.list_routable)
    dispatcher = Dispatcher(sm.put_target, abort_sleep_path=config.abort_sleep_path)
    policy = build_policy(config.policy, config)
    lock = OwnerLock(redis, config.lock_ttl_s)
    shell = BaselineShell(config, source, policy, dispatcher, redis, lock=lock,
                          decision_log=DecisionLog(config.log_dir, config.policy))
    server = make_http_server(shell, config.http_port)
    threading.Thread(target=server.serve_forever, name="bl-http", daemon=True).start()

    def _stop(signum, _frame) -> None:
        LOG.info("signal %s: stopping", signum)
        shell.stop()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    LOG.info("baseline scaler: arm=%s (policy=%s) dry_run=%s tick_s=%s abort_sleep_path=%s models=%s",
             shell.arm, config.policy, config.dry_run, config.tick_s, config.abort_sleep_path,
             sorted(config.models))
    try:
        shell.run()
    finally:
        lock.release()
        dispatcher.close()
        source.close()
        server.shutdown()


if __name__ == "__main__":
    main()
