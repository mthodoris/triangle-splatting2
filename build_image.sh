#!/bin/bash
# Build and push the cluster image for the current commit (fast update, notes/cluster.md).
# Usage (from anywhere):  bash /data/manousis/dev/bbtriangleSplat/combined/build_image.sh
# It prints the image name to put in the job YAMLs (Claude writes the YAMLs with it).
set -euo pipefail
cd "$(dirname "$0")"
REG=195.251.117.42:31000/textured-triangles
BASE=${BASE:-$REG:3ef9c33}
COMMIT=$(git rev-parse HEAD)
SHORT=${COMMIT:0:7}
# the image clones the code from GitHub, so the commit must be pushed there
git fetch -q origin
git branch -r --contains "$COMMIT" | grep -q . || { echo "commit $SHORT is not on GitHub yet (git push first)"; exit 1; }

echo "=== building $REG:$SHORT (code $COMMIT, from $BASE)"
docker build -f Dockerfile.update --build-arg BASE="$BASE" --build-arg COMMIT="$COMMIT" -t "$REG:$SHORT" .
echo "=== pushing"
docker push "$REG:$SHORT"
if curl -sk "https://${REG%%/*}/v2/textured-triangles/tags/list" | grep -q "\"$SHORT\""; then
  echo "=== OK: $REG:$SHORT is on the registry. You can submit the jobs that use it."
else
  echo "=== the push did not arrive on the registry; run this script again"; exit 1
fi
