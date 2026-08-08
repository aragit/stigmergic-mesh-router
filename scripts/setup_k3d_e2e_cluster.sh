#!/usr/bin/env bash
set -euo pipefail

CLUSTER_NAME="stigmergic-cluster"
NAMESPACE="default"
MONITORING_NAMESPACE="monitoring"
CHART_DIR="deploy/helm/stigmergic-mesh-router"
IMAGE_TAG="stigmergic-mesh-router:v1.0.0"

echo "==> Step 1: Checking required tool dependencies..."
for tool in k3d kubectl helm docker; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Error: Required CLI tool '$tool' is not installed." >&2
    exit 1
  fi
done
echo "✓ All prerequisite CLI tools detected."

echo "==> Step 2: Provisioning k3d Kubernetes cluster..."
if k3d cluster list "$CLUSTER_NAME" >/dev/null 2>&1; then
  echo "Deleting existing cluster '$CLUSTER_NAME' for a clean state..."
  k3d cluster delete "$CLUSTER_NAME"
fi

k3d cluster create "$CLUSTER_NAME" \
  --agents 2 \
  --port "8000:8000@loadbalancer" \
  --wait

kubectl cluster-info
echo "✓ k3d cluster '$CLUSTER_NAME' created and ready."

echo "==> Step 3: Building and importing local router Docker image..."
docker build -t "$IMAGE_TAG" .
k3d image import "$IMAGE_TAG" -c "$CLUSTER_NAME"
echo "✓ Image '$IMAGE_TAG' imported into k3d cluster nodes."

echo "==> Step 4: Installing Prometheus Stack and CRDs via Helm..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace "$MONITORING_NAMESPACE" \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
  --set prometheus.prometheusSpec.serviceMonitorNamespaceSelector.any=true \
  --set prometheus.service.type=ClusterIP \
  --wait

echo "Waiting for Prometheus pods to become ready..."
kubectl -n "$MONITORING_NAMESPACE" wait --for=condition=ready pod \
  -l app.kubernetes.io/name=prometheus --timeout=240s || \
  echo "⚠ Prometheus pods not confirmed ready via label selector; continuing."
echo "✓ Prometheus Operator & ServiceMonitor CRDs installed."

echo "==> Step 5: Deploying Prometheus Adapter for Custom Metrics API..."
PROM_SERVICE=""
for _ in $(seq 1 40); do
  PROM_SERVICE=$(kubectl get svc -n "$MONITORING_NAMESPACE" \
    -o jsonpath='{range .items[*]}{.metadata.name} {.spec.clusterIP} {.spec.ports[*].port}{"\n"}{end}' 2>/dev/null \
    | awk '$2!="None"{for(i=3;i<=NF;i++) if($i=="9090"){print $1; exit}}')
  [ -n "$PROM_SERVICE" ] && break
  sleep 3
done
if [ -z "$PROM_SERVICE" ]; then
  for cand in prometheus-kube-prometheus-stack-prometheus prometheus-prometheus prometheus; do
    if kubectl get svc -n "$MONITORING_NAMESPACE" "$cand" >/dev/null 2>&1; then
      PROM_SERVICE="$cand"
      break
    fi
  done
fi
if [ -z "$PROM_SERVICE" ]; then
  echo "Error: could not discover the Prometheus service in namespace '$MONITORING_NAMESPACE'." >&2
  exit 1
fi
echo "✓ Prometheus service resolved: ${PROM_SERVICE}.${MONITORING_NAMESPACE}.svc"

cat > /tmp/prometheus-adapter-values.yaml <<EOF
prometheus:
  url: http://${PROM_SERVICE}.${MONITORING_NAMESPACE}.svc
  port: 9090
rules:
  default: false
  custom:
    - seriesQuery: 'stigmergic_entropy_rate_total{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)_total"
        as: "\${1}"
      metricsQuery: 'sum(rate(<<.Series>>{<<.LabelMatchers>>}[2m])) by (<<.GroupBy>>)'
    - seriesQuery: 'agent_active_mesh_routes{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace: {resource: "namespace"}
          pod: {resource: "pod"}
      name:
        matches: "^(.*)"
        as: "\${1}"
      metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>}) by (<<.GroupBy>>)'
EOF

helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace "$MONITORING_NAMESPACE" \
  --create-namespace \
  --values /tmp/prometheus-adapter-values.yaml \
  --wait

echo "Waiting for the Custom Metrics APIService to become available..."
kubectl wait --for=condition=available apiservice v1beta1.custom.metrics.k8s.io --timeout=120s || true
echo "✓ Prometheus Adapter installed and Custom Metrics API registered."

echo "==> Step 6: Deploying Stigmergic Mesh Router Helm Chart..."
helm install stigmergic-mesh-router "$CHART_DIR" \
  --namespace "$NAMESPACE" \
  --create-namespace \
  --set image.repository="stigmergic-mesh-router" \
  --set image.tag="v1.0.0" \
  --set image.pullPolicy="IfNotPresent" \
  --set service.httpPort=8000 \
  --set service.type=LoadBalancer \
  --set autoscaling.enabled=true \
  --set serviceMonitor.enabled=true \
  --set serviceMonitor.namespace="$NAMESPACE" \
  --set prometheusAdapter.enabled=false

# The local dev image serves HTTP + /metrics + /health on a single port (8000),
# while the chart assumes 8080 / 9090 with /healthz and /ready probes. Reconcile
# the live deployment so pods become Ready and Prometheus scrapes /metrics:8000.
kubectl patch deployment stigmergic-mesh-router -n "$NAMESPACE" --type=strategic -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [
          {
            "name": "stigmergic-mesh-router",
            "livenessProbe":  {"httpGet": {"path": "/health", "port": "http"}, "initialDelaySeconds": 15, "periodSeconds": 10, "timeoutSeconds": 2, "failureThreshold": 3},
            "readinessProbe": {"httpGet": {"path": "/health", "port": "http"}, "initialDelaySeconds": 5,  "periodSeconds": 5,  "timeoutSeconds": 2, "failureThreshold": 3}
          }
        ]
      }
    }
  }
}
'

kubectl patch servicemonitor stigmergic-mesh-router -n "$NAMESPACE" --type=merge -p '
{
  "spec": {
    "endpoints": [
      {
        "port": "http",
        "path": "/metrics",
        "interval": "15s",
        "relabelings": [
          {"sourceLabels": ["__meta_kubernetes_namespace"], "targetLabel": "namespace"},
          {"sourceLabels": ["__meta_kubernetes_pod_name"],  "targetLabel": "pod"}
        ]
      }
    ]
  }
}
'

echo "Waiting for deployment rollout..."
kubectl rollout status deployment/stigmergic-mesh-router -n "$NAMESPACE" --timeout=180s
echo "✓ Stigmergic Mesh Router deployed successfully."

echo "==> Step 7: Verifying Custom Metrics API & HPA..."
echo "Waiting for Prometheus to scrape metrics and expose them via the Adapter (entropy rate is a 2-minute rate)..."
DEADLINE=$((SECONDS + 300))
HAVE_ENTROPY=0
HAVE_ROUTES=0
while [ "$SECONDS" -lt "$DEADLINE" ]; do
  API_VERSION="v1beta1"
  API=$(kubectl get --raw "/apis/custom.metrics.k8s.io/${API_VERSION}" 2>/dev/null || true)
  if [ -z "$API" ]; then
    API_VERSION="v1beta2"
    API=$(kubectl get --raw "/apis/custom.metrics.k8s.io/${API_VERSION}" 2>/dev/null || true)
  fi
  if printf '%s' "$API" | grep -q "stigmergic_entropy_rate"; then HAVE_ENTROPY=1; fi
  if printf '%s' "$API" | grep -q "agent_active_mesh_routes"; then HAVE_ROUTES=1; fi
  if [ "$HAVE_ENTROPY" -eq 1 ] && [ "$HAVE_ROUTES" -eq 1 ]; then break; fi
  sleep 10
done

echo "--- Custom Metrics API Discovery ---"
[ "$HAVE_ENTROPY" -eq 1 ] && echo "✓ stigmergic_entropy_rate is exposed via custom.metrics.k8s.io" \
  || echo "⚠ stigmergic_entropy_rate not exposed yet (needs ~2m of counter history in Prometheus)"
[ "$HAVE_ROUTES" -eq 1 ] && echo "✓ agent_active_mesh_routes is exposed via custom.metrics.k8s.io" \
  || echo "⚠ agent_active_mesh_routes not exposed yet"

echo "--- Live custom metric sample (agent_active_mesh_routes) ---"
ROUTER_POD=$(kubectl get pods -n "$NAMESPACE" -l app.kubernetes.io/name=stigmergic-mesh-router -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true)
if [ -n "$ROUTER_POD" ]; then
  kubectl get --raw "/apis/custom.metrics.k8s.io/${API_VERSION}/namespaces/${NAMESPACE}/pods/${ROUTER_POD}/agent_active_mesh_routes" 2>/dev/null || echo "(no value yet)"
fi

echo "--- HorizontalPodAutoscaler Status ---"
kubectl get hpa stigmergic-mesh-router -n "$NAMESPACE" -o wide

echo ""
echo "========================================================================="
echo " E2E Cluster Setup Complete!"
echo "========================================================================="
echo "Router (OpenAI-compatible):  http://localhost:8000   (LoadBalancer svc)"
echo "Prometheus UI:               kubectl port-forward -n $MONITORING_NAMESPACE svc/${PROM_SERVICE} 9090:9090"
echo ""
echo "Generate load to drive the custom metrics above the HPA targets, then watch:"
echo "  kubectl get hpa stigmergic-mesh-router -n $NAMESPACE -w"
echo ""
echo "To clean up and delete the cluster:"
echo "  k3d cluster delete $CLUSTER_NAME"
echo "========================================================================="
