from tre_sm.gpu_truth import NullGpuTruth, RedisGpuTruth


class FakeRedis:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def get(self, key):
        return self.values.get(key)


def test_null_gpu_truth_returns_none():
    truth = NullGpuTruth()

    assert truth.used_mib(node="node-a", gpu_id=0, gpu_uuid="GPU-0") is None


def test_redis_gpu_truth_reads_node_payload_by_uuid():
    redis = FakeRedis(
        {
            "tre:gpu_truth:node-a": (
                b'{"gpus":[{"uuid":"GPU-0","used_mib":123},'
                b'{"uuid":"GPU-1","used_mib":456}]}'
            )
        }
    )
    truth = RedisGpuTruth(redis)

    assert truth.used_mib(node="node-a", gpu_id=1, gpu_uuid="GPU-1") == 456
    assert truth.used_mib(node="node-a", gpu_id=2, gpu_uuid="GPU-2") is None


def test_redis_gpu_truth_returns_none_for_bad_or_missing_payload():
    truth = RedisGpuTruth(FakeRedis({"tre:gpu_truth:node-a": b"not-json"}))

    assert truth.used_mib(node="node-a", gpu_id=0, gpu_uuid="GPU-0") is None
    assert truth.used_mib(node="node-b", gpu_id=0, gpu_uuid="GPU-0") is None
