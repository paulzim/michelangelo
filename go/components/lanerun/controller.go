// Package lanerun implements a Kubernetes controller for the LaneRun CRD, the
// "Michelangelo Pit Stop" KubeCon demo's per-visitor race configuration.
//
// The controller drives a LaneRun through DeployPhase (SPEC_WRITTEN ->
// CONTROLLER_SYNCED -> [ADVISOR_QUERIED] -> READY), matching the deploy_phase
// contract booth-api (a separate repo, not yet implemented) surfaces to the
// tablet/overlay frontends. When Spec.Mode is RECOMMENDED, the controller
// itself calls the Pit Crew Advisor's InferenceServer through the shared
// gateway to fill in the recommended speed cap and caution buffer.
package lanerun

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"reflect"
	"time"

	"go.uber.org/zap"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/michelangelo-ai/michelangelo/go/api"
	apiHandler "github.com/michelangelo-ai/michelangelo/go/api/handler"
	"github.com/michelangelo-ai/michelangelo/go/api/utils"
	maconfig "github.com/michelangelo-ai/michelangelo/go/base/config"
	conditionUtils "github.com/michelangelo-ai/michelangelo/go/base/conditions/utils"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
)

const (
	// reconcileInterval defines how frequently non-terminal LaneRuns are
	// reconciled, including retrying a failed advisor query.
	reconcileInterval = 5 * time.Second

	// defaultAdvisorTimeout bounds the advisor HTTP call when Config.AdvisorTimeout is unset.
	defaultAdvisorTimeout = 5 * time.Second

	// conditionTypeAdvisorQueried reports the health of the most recent advisor query.
	conditionTypeAdvisorQueried = "AdvisorQueried"

	// advisorInputName/advisorOutputName must match the ModelSchema the
	// pitstop_advisor pipeline packages its model with.
	advisorInputName  = "features"
	advisorOutputName = "settings"
)

// Reconciler implements the controller-runtime Reconciler interface for LaneRun resources.
type Reconciler struct {
	api.Handler
	logger            *zap.Logger
	apiHandlerFactory apiHandler.Factory
	kubeClient        client.Client
	gatewayConfig     maconfig.InferenceServerConfig
	httpClient        *http.Client
	config            Config
}

// NewReconciler constructs a Reconciler with required dependencies.
func NewReconciler(
	apiHandlerFactory apiHandler.Factory,
	logger *zap.Logger,
	gatewayConfig maconfig.InferenceServerConfig,
	cfg Config,
) *Reconciler {
	timeout := cfg.AdvisorTimeout
	if timeout <= 0 {
		timeout = defaultAdvisorTimeout
	}
	return &Reconciler{
		apiHandlerFactory: apiHandlerFactory,
		logger:            logger,
		gatewayConfig:     gatewayConfig,
		httpClient:        &http.Client{Timeout: timeout},
		config:            cfg,
	}
}

// Reconcile drives a LaneRun through its DeployPhase state machine.
func (r *Reconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := r.logger.With(zap.String("namespace-name", req.NamespacedName.String()))
	laneRun := &v2pb.LaneRun{}
	if err := r.Get(ctx, req.Namespace, req.Name, &metav1.GetOptions{}, laneRun); err != nil {
		// The API handler surfaces not-found as a gRPC status error, so use
		// utils.IsNotFoundError (handles both gRPC and k8s-typed errors) rather
		// than client.IgnoreNotFound (k8s-typed only).
		if utils.IsNotFoundError(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}
	if !laneRun.GetDeletionTimestamp().IsZero() {
		return ctrl.Result{}, nil
	}
	original := laneRun.DeepCopy()
	logger.Info("Reconciling LaneRun", zap.String("deployPhase", laneRun.Status.DeployPhase.String()))

	switch laneRun.Status.DeployPhase {
	case v2pb.DEPLOY_PHASE_INVALID, v2pb.DEPLOY_PHASE_SPEC_WRITTEN:
		if err := validateSpec(laneRun.Spec); err != nil {
			logger.Warn("LaneRun spec invalid; will retry", zap.Error(err))
			break
		}
		laneRun.Status.DeployPhase = v2pb.DEPLOY_PHASE_CONTROLLER_SYNCED
	case v2pb.DEPLOY_PHASE_CONTROLLER_SYNCED:
		if laneRun.Spec.Mode == v2pb.LANE_RUN_MODE_RECOMMENDED {
			r.queryAdvisor(ctx, logger, laneRun)
		} else {
			laneRun.Status.DeployPhase = v2pb.DEPLOY_PHASE_READY
		}
	case v2pb.DEPLOY_PHASE_ADVISOR_QUERIED:
		laneRun.Status.DeployPhase = v2pb.DEPLOY_PHASE_READY
	case v2pb.DEPLOY_PHASE_READY:
		// Terminal; nothing to do.
	}

	return r.updateStatus(ctx, laneRun, original, logger)
}

// validateSpec rejects LaneRuns missing the fields required to progress past SPEC_WRITTEN.
func validateSpec(spec v2pb.LaneRunSpec) error {
	if spec.Lane == v2pb.LANE_INVALID {
		return fmt.Errorf("spec.lane is required")
	}
	if spec.Mode == v2pb.LANE_RUN_MODE_INVALID {
		return fmt.Errorf("spec.mode is required")
	}
	return nil
}

// queryAdvisor calls the Pit Crew Advisor and records the outcome on laneRun.Status.
// A failed or timed-out query is recorded as a false condition rather than
// returned as a reconcile error, so DeployPhase simply stays at
// CONTROLLER_SYNCED and retries on the next reconcile instead of backing off.
func (r *Reconciler) queryAdvisor(ctx context.Context, logger *zap.Logger, laneRun *v2pb.LaneRun) {
	cond := conditionUtils.GetCondition(conditionTypeAdvisorQueried, laneRun.Status.Conditions)
	if cond == nil {
		cond = &apipb.Condition{Type: conditionTypeAdvisorQueried, Status: apipb.CONDITION_STATUS_UNKNOWN}
		laneRun.Status.Conditions = append(laneRun.Status.Conditions, cond)
	}

	speedCapCms, cautionBufferCm, err := r.callAdvisor(ctx, laneRun)
	if err != nil {
		logger.Warn("Pit Crew Advisor query failed; will retry", zap.Error(err))
		laneRun.Status.AdvisorHealthy = false
		conditionUtils.GenerateFalseCondition(cond, err.Error(), "AdvisorQueryFailed")
		return
	}

	laneRun.Status.AdvisorHealthy = true
	laneRun.Status.RecommendedSpeedCapCms = speedCapCms
	laneRun.Status.RecommendedCautionBufferCm = cautionBufferCm
	conditionUtils.GenerateTrueCondition(cond)
	laneRun.Status.DeployPhase = v2pb.DEPLOY_PHASE_ADVISOR_QUERIED
}

// kserveTensor is a single named tensor in the KServe v2 inference protocol,
// used for both the request's "inputs" and the response's "outputs".
type kserveTensor struct {
	Name     string    `json:"name"`
	Shape    []int     `json:"shape"`
	Datatype string    `json:"datatype"`
	Data     []float32 `json:"data"`
}

type kserveInferRequest struct {
	Inputs []kserveTensor `json:"inputs"`
}

type kserveInferResponse struct {
	Outputs []kserveTensor `json:"outputs"`
}

// callAdvisor sends the LaneRun's context to the Pit Crew Advisor's /infer
// endpoint and returns its recommended speed_cap_cms/caution_buffer_cm.
//
// The feature vector is deliberately minimal: [laneFeature, trackGrip]. There
// is no live telemetry yet (car-agent, the source of real track/lap signal,
// hasn't been built), so trackGrip is a fixed nominal placeholder that the
// pitstop_advisor training data also treats as a feature — this keeps the
// train/serve contract real and demoable now, and is the one line to change
// once car-agent supplies live values.
func (r *Reconciler) callAdvisor(ctx context.Context, laneRun *v2pb.LaneRun) (int32, int32, error) {
	if r.config.AdvisorInferenceServerName == "" || r.config.AdvisorDeploymentName == "" {
		return 0, 0, fmt.Errorf("advisor inference server/deployment not configured")
	}

	gatewayURL, err := r.resolveGatewayEndpoint(ctx)
	if err != nil {
		return 0, 0, fmt.Errorf("resolve gateway endpoint: %w", err)
	}

	const nominalTrackGrip = 0.75
	reqBody := kserveInferRequest{
		Inputs: []kserveTensor{
			{
				Name:     advisorInputName,
				Shape:    []int{1, 2},
				Datatype: "FP32",
				Data:     []float32{laneFeatureValue(laneRun.Spec.Lane), nominalTrackGrip},
			},
		},
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return 0, 0, fmt.Errorf("marshal infer request: %w", err)
	}

	url := fmt.Sprintf("%s/%s/%s/infer", gatewayURL, r.config.AdvisorInferenceServerName, r.config.AdvisorDeploymentName)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return 0, 0, fmt.Errorf("create infer request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := r.httpClient.Do(httpReq)
	if err != nil {
		return 0, 0, fmt.Errorf("call advisor infer endpoint %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return 0, 0, fmt.Errorf("advisor infer endpoint %s returned %s: %s", url, resp.Status, string(body))
	}

	var inferResp kserveInferResponse
	if err := json.NewDecoder(resp.Body).Decode(&inferResp); err != nil {
		return 0, 0, fmt.Errorf("decode infer response: %w", err)
	}
	for _, out := range inferResp.Outputs {
		if out.Name == advisorOutputName && len(out.Data) >= 2 {
			return int32(out.Data[0]), int32(out.Data[1]), nil
		}
	}
	return 0, 0, fmt.Errorf("advisor infer response missing %q output", advisorOutputName)
}

// laneFeatureValue encodes Lane as a 0/1 model feature. The enum's own
// ordinal (LANE_A=1, LANE_B=2) is a wire-format detail, not a meaningful
// magnitude, so it is not used directly.
func laneFeatureValue(lane v2pb.Lane) float32 {
	if lane == v2pb.LANE_B {
		return 1
	}
	return 0
}

// resolveGatewayEndpoint returns the base URL of the in-cluster gateway
// Service, mirroring the NodePort + Node-InternalIP resolution
// go/components/inferenceserver/endpoints/provider/k8s.go performs for
// remote cluster targets — simplified here because the LaneRun controller
// only ever talks to the gateway in its own cluster.
func (r *Reconciler) resolveGatewayEndpoint(ctx context.Context) (string, error) {
	gw := r.gatewayConfig.Gateway
	svc := &corev1.Service{}
	key := types.NamespacedName{Name: gw.ServiceName, Namespace: gw.ServiceNamespace}
	if err := r.kubeClient.Get(ctx, key, svc); err != nil {
		return "", fmt.Errorf("get gateway service %s/%s: %w", gw.ServiceNamespace, gw.ServiceName, err)
	}

	var nodePort int32
	for _, port := range svc.Spec.Ports {
		if port.Name == gw.PortName {
			nodePort = port.NodePort
			break
		}
	}
	if nodePort == 0 {
		return "", fmt.Errorf("gateway service %s/%s has no NodePort on port %q", gw.ServiceNamespace, gw.ServiceName, gw.PortName)
	}

	nodes := &corev1.NodeList{}
	if err := r.kubeClient.List(ctx, nodes); err != nil {
		return "", fmt.Errorf("list nodes: %w", err)
	}
	for _, node := range nodes.Items {
		for _, addr := range node.Status.Addresses {
			if addr.Type == corev1.NodeInternalIP && addr.Address != "" {
				return fmt.Sprintf("http://%s:%d", addr.Address, nodePort), nil
			}
		}
	}
	return "", fmt.Errorf("no node reported an InternalIP")
}

// updateStatus persists laneRun.Status when it changed and schedules a
// requeue for non-terminal phases.
func (r *Reconciler) updateStatus(ctx context.Context, laneRun *v2pb.LaneRun, original *v2pb.LaneRun, logger *zap.Logger) (ctrl.Result, error) {
	result := ctrl.Result{}
	if laneRun.Status.DeployPhase != v2pb.DEPLOY_PHASE_READY {
		result = ctrl.Result{RequeueAfter: reconcileInterval}
	}
	if !reflect.DeepEqual(original.Status, laneRun.Status) {
		logger.Info("LaneRun status updated", zap.String("deployPhase", laneRun.Status.DeployPhase.String()))
		if err := r.UpdateStatus(ctx, laneRun, &metav1.UpdateOptions{}); err != nil {
			return result, fmt.Errorf("update LaneRun status for %s/%s: %w", laneRun.Namespace, laneRun.Name, err)
		}
	}
	return result, nil
}

// Register sets up the LaneRun controller with the controller-runtime manager.
func (r *Reconciler) Register(mgr ctrl.Manager) error {
	handler, err := r.apiHandlerFactory.GetAPIHandler(mgr.GetClient())
	if err != nil {
		return err
	}
	r.Handler = handler
	r.kubeClient = mgr.GetClient()
	return ctrl.NewControllerManagedBy(mgr).
		For(&v2pb.LaneRun{}).
		Complete(r)
}
