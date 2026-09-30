#!/usr/bin/env python3
"""Strip server-side fields from backed-up Kubernetes objects so they can be
``kubectl replace``-d / ``create``-d again.

  strip_obj.py <kind> <ns> <name> <list-yaml>    one object of a List
  strip_obj.py - - - <file>                      a single object file
  strip_obj.py --models <node> - - <list-yaml>   List of the model Deployments of <node>
  strip_obj.py --all <list-yaml>                 every object of a List (e.g. Envoy policies)
"""
import sys

import yaml

DROP = ("resourceVersion", "uid", "creationTimestamp", "generation", "managedFields", "selfLink")
#: Annotations the apiserver / kubectl add; replaying them would pin a stale object.
DROP_ANNOTATIONS = ("kubectl.kubernetes.io/last-applied-configuration", "deployment.kubernetes.io/revision")


def clean(o):
    meta = o.get("metadata") or {}
    for k in DROP:
        meta.pop(k, None)
    annotations = meta.get("annotations") or {}
    for k in DROP_ANNOTATIONS:
        annotations.pop(k, None)
    if "annotations" in meta and not annotations:
        meta.pop("annotations")
    o.pop("status", None)
    return o


def dump_list(items):
    yaml.safe_dump({"apiVersion": "v1", "kind": "List", "items": items}, sys.stdout, sort_keys=False)


def main(a):
    if not a:
        raise SystemExit(__doc__)
    if a[0] == "--models":
        node, src = a[1], a[4]
        d = yaml.safe_load(open(src))
        dump_list([clean(i) for i in d["items"]
                   if i["kind"] == "Deployment" and i["metadata"]["labels"].get("tre.aibrix.io/node") == node])
    elif a[0] == "--all":
        dump_list([clean(i) for i in (yaml.safe_load(open(a[1])) or {}).get("items") or []])
    elif a[0] == "-":
        yaml.safe_dump(clean(yaml.safe_load(open(a[3]))), sys.stdout, sort_keys=False)
    else:
        kind, ns, name, src = a
        items = yaml.safe_load(open(src))["items"]
        [o] = [i for i in items if i["kind"] == kind and i["metadata"]["name"] == name
               and i["metadata"].get("namespace") == ns]
        yaml.safe_dump(clean(o), sys.stdout, sort_keys=False)


if __name__ == "__main__":
    main(sys.argv[1:])
