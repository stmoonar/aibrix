#!/usr/bin/env bash
# Rebuild the TRE gateway-plugins image the way 20260704-0d869b49-nozmq2 was built.
#
# Recipe recovered from that image (2026-09-24): `go version -m /gateway-plugins` reports
# go1.22.6, -tags=nozmq, -trimpath, CGO_ENABLED=0, vcs.modified=false; its 13 layers are
# the 12 layers of aibrix/gateway-plugins:latest (sha256:db7f208f301a..., the distroless
# upstream build of 2026-04-25) plus one COPY of the binary, USER 65532, ENTRYPOINT
# /gateway-plugins. No Dockerfile for it was ever committed; this script is that recipe.
#
# Usage (on 76, from a CLEAN checkout - the binary records vcs.modified):
#   tre/deploy/scripts/build_gateway_plugins_nozmq.sh [tag]
# Default tag: aibrix/gateway-plugins:<YYYYMMDD>-<HEAD short sha>-nozmq2
# Long build: run under nohup, e.g.
#   nohup tre/deploy/scripts/build_gateway_plugins_nozmq.sh > /tmp/build-gwp.log 2>&1 &
set -euo pipefail

REPO="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
cd "$REPO"

BASE_IMAGE="${BASE_IMAGE:-aibrix/gateway-plugins:latest}"
BASE_ID_EXPECTED="${BASE_ID_EXPECTED:-sha256:db7f208f301a148257af593a75a105c82f33b79f5d55cbbe497aeeaebb2d914f}"
GO_TOOLCHAIN="${GO_TOOLCHAIN:-go1.22.6}"
SHA="$(git rev-parse --short=8 HEAD)"
TAG="${1:-aibrix/gateway-plugins:$(date +%Y%m%d)-${SHA}-nozmq2}"

if [ -n "$(git status --porcelain -- pkg cmd go.mod go.sum)" ]; then
  echo "refusing to build: pkg/ cmd/ go.mod go.sum have uncommitted changes" >&2
  exit 1
fi

base_id="$(docker image inspect "$BASE_IMAGE" --format '{{.Id}}')"
if [ "$base_id" != "$BASE_ID_EXPECTED" ]; then
  echo "base image $BASE_IMAGE is $base_id, expected $BASE_ID_EXPECTED" >&2
  echo "(override BASE_ID_EXPECTED only if you mean to change the base)" >&2
  exit 1
fi

CTX="$(mktemp -d /tmp/gwp-nozmq-build.XXXXXX)"
trap 'rm -rf "$CTX"' EXIT

export PATH="$PATH:/usr/local/go/bin"
echo "[build] go build ($GO_TOOLCHAIN, -tags=nozmq, CGO_ENABLED=0, -trimpath) at $SHA"
GOTOOLCHAIN="$GO_TOOLCHAIN" GOPROXY="${GOPROXY:-off}" CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
  go build -tags=nozmq -trimpath -o "$CTX/gateway-plugins" cmd/plugins/main.go

cat > "$CTX/Dockerfile" <<EOF
FROM ${BASE_IMAGE}
COPY --chown=65532:65532 gateway-plugins /gateway-plugins
USER 65532:65532
ENTRYPOINT ["/gateway-plugins"]
EOF

echo "[build] docker build -t $TAG"
docker build -t "$TAG" "$CTX"
docker image inspect "$TAG" --format '{{.Id}} {{.Created}}'
echo "[build] done: $TAG"
