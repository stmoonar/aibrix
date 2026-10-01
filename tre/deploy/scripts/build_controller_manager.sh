#!/usr/bin/env bash
# Offline rebuild of the AIBrix controller-manager image (the podautoscaler that drives the
# APA baseline arm), equivalent to build/container/Dockerfile.
#
# build/container/Dockerfile needs network (`go mod download` via GOPROXY). This script does
# the same build offline: go1.22 container + the host Go module cache (GOPROXY=off),
# CGO_ENABLED=0, then packages the binary on the same distroless base
# (gcr.io/distroless/static:nonroot) with the same WORKDIR / USER / ENTRYPOINT.
# Unlike the Dockerfile it builds the package path (./cmd/controllers) so the binary is
# stamped with vcs.revision / vcs.modified; check with `go version -m /manager`.
#
# Usage (from a CLEAN, non-worktree clone; a git worktree records no vcs.* info):
#   tre/deploy/scripts/build_controller_manager.sh [tag]
# Default tag: aibrix/controller-manager:<YYYYMMDD>-<HEAD short sha>
# Env: GO_IMAGE (golang:1.22), BASE_IMAGE (gcr.io/distroless/static:nonroot),
#      GOMODCACHE_HOST (/root/go/pkg/mod)
# Long build: nohup tre/deploy/scripts/build_controller_manager.sh > /tmp/build-cm.log 2>&1 &
set -euo pipefail

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO"

GO_IMAGE="${GO_IMAGE:-golang:1.22}"
BASE_IMAGE="${BASE_IMAGE:-gcr.io/distroless/static:nonroot}"
GOMODCACHE_HOST="${GOMODCACHE_HOST:-/root/go/pkg/mod}"
SHA="$(git rev-parse --short=8 HEAD)"
TAG="${1:-aibrix/controller-manager:$(date +%Y%m%d)-${SHA}}"

if [ -n "$(git status --porcelain -- pkg cmd api go.mod go.sum)" ]; then
  echo "refusing to build: pkg/ cmd/ api/ go.mod go.sum have uncommitted changes" >&2
  exit 1
fi
docker image inspect "$BASE_IMAGE" >/dev/null

CTX="$(mktemp -d /tmp/cm-build.XXXXXX)"
trap 'rm -rf "$CTX"' EXIT

echo "[build] go build ($GO_IMAGE, CGO_ENABLED=0, GOPROXY=off) at $SHA"
docker run --rm -v "$REPO":/src -w /src -v "$GOMODCACHE_HOST":/go/pkg/mod -v "$CTX":/out \
  -e GOPROXY=off -e GOFLAGS=-mod=mod -e CGO_ENABLED=0 -e GOOS=linux -e GOARCH=amd64 \
  "$GO_IMAGE" sh -c 'git config --global --add safe.directory /src &&
    go build -a -o /out/manager ./cmd/controllers && go version -m /out/manager | grep -E "go1|vcs"'

cat > "$CTX/Dockerfile" <<EOF
FROM ${BASE_IMAGE}
WORKDIR /
COPY manager .
USER 65532:65532
ENTRYPOINT ["/manager"]
EOF

echo "[build] docker build -t $TAG"
docker build -t "$TAG" "$CTX"
docker image inspect "$TAG" --format '{{.Id}} {{.Created}}'
echo "[build] done: $TAG"
