#!/usr/bin/env python3
"""Replace registry.yaml of ConfigMap tre-v2-registry with FILE, the way params_set.py --apply does
(backup of the live CM first, resourceVersion lock -> 409 on a concurrent edit). WRITES THE CLUSTER.

  cm_replace.py FILE --backup-dir DIR [--expect-rv RV] [--expect-id ID]
--expect-rv: refuse unless the live CM is still at RV (the version FILE was built from).
--expect-id: refuse unless the live registry content id (arm_registry.registry_id) is ID.
"""
import argparse, json, os, sys
from pathlib import Path

TRE = Path(os.environ.get("TRE_DIR", "/data/nfs_shared_data/xxy/aibrix/tre"))
sys.path[:0] = [str(TRE / "common"), str(TRE / "deploy"), str(TRE / "deploy" / "scripts" / "release"),
                str(TRE / "eval" / "runner")]
import params_set as PS  # noqa: E402
import arm_registry as AR  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file"); ap.add_argument("--backup-dir", required=True)
    ap.add_argument("--expect-rv"); ap.add_argument("--expect-id")
    a = ap.parse_args()
    new = Path(a.file).read_text()
    cm = PS.read_configmap()
    rv = cm["metadata"].get("resourceVersion"); live = (cm.get("data") or {}).get(PS.DATA_KEY, "")
    if (cm["metadata"].get("annotations") or {}).get(AR.ANNOTATION):
        sys.exit("REFUSED: arm registry perturbation annotation present")
    if a.expect_rv and rv != a.expect_rv:
        sys.exit(f"REFUSED: live resourceVersion {rv} != expected {a.expect_rv} (rebuild the file)")
    if a.expect_id and AR.registry_id(live) != a.expect_id:
        sys.exit(f"REFUSED: live registry id {AR.registry_id(live)} != expected {a.expect_id}")
    errs = PS.rp.new_validation_errors(live, new)
    if errs:
        sys.exit(f"REFUSED: Registry.validate() errors: {errs}")
    print("backup:", PS.backup(cm, a.backup_dir))
    PS.kubectl(["replace", "-f", "-"], stdin=json.dumps(PS.replacement(cm, new)))
    print(f"replaced (was rv {rv}); id {AR.registry_id(live)} -> {AR.registry_id(new)}; "
          f"params-hash {PS.rp.params_hash(new)}")


if __name__ == "__main__":
    main()
