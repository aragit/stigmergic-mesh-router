#!/usr/bin/env bash
set -euo pipefail

CHART_DIR="deploy/helm/stigmergic-mesh-router"

echo "==> Step 1: Linting Helm Chart..."
helm lint "${CHART_DIR}"

echo "==> Step 2: Rendering Templates (Dry-Run)..."
helm template test-release "${CHART_DIR}" \
  --set autoscaling.enabled=true \
  --set prometheusAdapter.enabled=true > /tmp/rendered_stigmergic_manifests.yaml

echo "==> Step 3: Validating HPA autoscaling/v2 manifest structure..."
grep -A 20 "kind: HorizontalPodAutoscaler" /tmp/rendered_stigmergic_manifests.yaml | grep -q "stigmergic_entropy_rate"
if [ $? -eq 0 ]; then
    echo "✓ Custom Metric 'stigmergic_entropy_rate' verified in HPA spec."
else
    echo "✗ Failed to find custom metric in HPA spec!"
    exit 1
fi

echo "==> Step 4: Schema validation with kubeconform (if installed)..."
if command -v kubeconform &> /dev/null; then
    kubeconform -strict -kubernetes-version 1.30.0 /tmp/rendered_stigmergic_manifests.yaml
    echo "✓ Kubernetes Manifest Schema Validation passed."
else
    echo "⚠ kubeconform not installed. Skipping strict Kubernetes schema check."
fi

echo "✓ Phase 10 Validation Complete!"
