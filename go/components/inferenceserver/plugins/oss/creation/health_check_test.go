package creation

import (
	"context"
	"errors"
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/backends"
	backendsmocks "github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/backends/backendsmocks"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
)

// createHealthCheckTestRegistry creates a registry with the mock backend registered for Triton.
func createHealthCheckTestRegistry(mockBackend *backendsmocks.MockBackend) *backends.Registry {
	registry := backends.NewRegistry()
	registry.Register(v2pb.BACKEND_TYPE_TRITON, mockBackend)
	return registry
}

func TestHealthCheckActor_Retrieve(t *testing.T) {
	tests := []struct {
		name                   string
		clusterTargets         []*v2pb.ClusterTarget
		clientFactoryErrors    map[string]error
		setupMocks             func(*backendsmocks.MockBackend)
		registryHasBackend     bool
		expectedStatus         apipb.ConditionStatus
		expectedReason         string
		expectedMessage        string
		expectedClusterStatues []*v2pb.ClusterTargetStatus
	}{
		{
			name:                   "no clusters - vacuously healthy",
			clusterTargets:         nil,
			setupMocks:             func(_ *backendsmocks.MockBackend) {},
			registryHasBackend:     true,
			expectedStatus:         apipb.CONDITION_STATUS_TRUE,
			expectedClusterStatues: []*v2pb.ClusterTargetStatus{},
		},
		{
			name:           "server is healthy",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mockBackend *backendsmocks.MockBackend) {
				mockBackend.EXPECT().IsHealthy(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").Return(true, nil)
			},
			registryHasBackend: true,
			expectedStatus:     apipb.CONDITION_STATUS_TRUE,
			expectedClusterStatues: []*v2pb.ClusterTargetStatus{
				{ClusterId: "c1", State: v2pb.INFERENCE_SERVER_STATE_SERVING},
			},
		},
		{
			name:           "server is not healthy",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mockBackend *backendsmocks.MockBackend) {
				mockBackend.EXPECT().IsHealthy(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").Return(false, nil)
			},
			registryHasBackend: true,
			expectedStatus:     apipb.CONDITION_STATUS_FALSE,
			expectedMessage:    "HealthCheckFailed",
			expectedReason:     "c1: not healthy",
			expectedClusterStatues: []*v2pb.ClusterTargetStatus{
				{ClusterId: "c1", State: v2pb.INFERENCE_SERVER_STATE_CREATING},
			},
		},
		{
			name:           "health check returns error",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mockBackend *backendsmocks.MockBackend) {
				mockBackend.EXPECT().IsHealthy(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").Return(false, errors.New("connection timeout"))
			},
			registryHasBackend: true,
			expectedStatus:     apipb.CONDITION_STATUS_FALSE,
			expectedMessage:    "HealthCheckFailed",
			expectedReason:     "c1: connection timeout",
			expectedClusterStatues: []*v2pb.ClusterTargetStatus{
				{ClusterId: "c1", State: v2pb.INFERENCE_SERVER_STATE_CREATING, Message: "connection timeout"},
			},
		},
		{
			name:                "GetClient errors for single cluster",
			clusterTargets:      []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			clientFactoryErrors: map[string]error{"c1": errors.New("auth refused")},
			setupMocks:          func(_ *backendsmocks.MockBackend) {},
			registryHasBackend:  true,
			expectedStatus:      apipb.CONDITION_STATUS_FALSE,
			expectedMessage:     "HealthCheckFailed",
			expectedReason:      "c1: client error: auth refused",
			expectedClusterStatues: []*v2pb.ClusterTargetStatus{
				{ClusterId: "c1", State: v2pb.INFERENCE_SERVER_STATE_CREATING, Message: "auth refused"},
			},
		},
		{
			name:               "backend not in registry",
			clusterTargets:     []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks:         func(_ *backendsmocks.MockBackend) {},
			registryHasBackend: false,
			expectedStatus:     apipb.CONDITION_STATUS_FALSE,
			expectedMessage:    "BackendNotFound",
			expectedReason:     "Failed to get backend: backend not found for type: BACKEND_TYPE_TRITON",
		},
		{
			name: "mixed: serving + not-healthy + IsHealthy err + client err",
			clusterTargets: []*v2pb.ClusterTarget{
				{ClusterId: "c-ok"},
				{ClusterId: "c-unhealthy"},
				{ClusterId: "c-err"},
				{ClusterId: "c-noclient"},
			},
			clientFactoryErrors: map[string]error{"c-noclient": errors.New("no token")},
			setupMocks: func(mockBackend *backendsmocks.MockBackend) {
				gomock.InOrder(
					mockBackend.EXPECT().IsHealthy(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").Return(true, nil),
					mockBackend.EXPECT().IsHealthy(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").Return(false, nil),
					mockBackend.EXPECT().IsHealthy(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").Return(false, errors.New("dial timeout")),
				)
			},
			registryHasBackend: true,
			expectedStatus:     apipb.CONDITION_STATUS_FALSE,
			expectedMessage:    "HealthCheckFailed",
			expectedReason:     "c-unhealthy: not healthy; c-err: dial timeout; c-noclient: client error: no token",
			expectedClusterStatues: []*v2pb.ClusterTargetStatus{
				{ClusterId: "c-ok", State: v2pb.INFERENCE_SERVER_STATE_SERVING},
				{ClusterId: "c-unhealthy", State: v2pb.INFERENCE_SERVER_STATE_CREATING},
				{ClusterId: "c-err", State: v2pb.INFERENCE_SERVER_STATE_CREATING, Message: "dial timeout"},
				{ClusterId: "c-noclient", State: v2pb.INFERENCE_SERVER_STATE_CREATING, Message: "no token"},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mockBackend := backendsmocks.NewMockBackend(ctrl)
			registry := backends.NewRegistry()
			if tt.registryHasBackend {
				registry = createHealthCheckTestRegistry(mockBackend)
			}

			tt.setupMocks(mockBackend)

			factory := newClientFactoryDispatching(ctrl, tt.clientFactoryErrors)
			actor := NewHealthCheckActor(factory, registry, zap.NewNop())

			resource := &v2pb.InferenceServer{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-server",
					Namespace: "test-namespace",
				},
				Spec: v2pb.InferenceServerSpec{
					BackendType:    v2pb.BACKEND_TYPE_TRITON,
					ClusterTargets: tt.clusterTargets,
				},
			}

			condition := &apipb.Condition{
				Type: "TritonHealthCheck",
			}

			result, err := actor.Retrieve(context.Background(), resource, condition)

			require.NoError(t, err)
			assert.Equal(t, tt.expectedStatus, result.Status)
			assert.Equal(t, tt.expectedReason, result.Reason)
			assert.Equal(t, tt.expectedMessage, result.Message)
			assert.Equal(t, "TritonHealthCheck", result.Type)

			// Status.ClusterStatuses is only populated when the backend is found.
			if tt.registryHasBackend {
				require.Equal(t, len(tt.expectedClusterStatues), len(resource.Status.ClusterStatuses))
				for i, want := range tt.expectedClusterStatues {
					got := resource.Status.ClusterStatuses[i]
					assert.Equal(t, want.GetClusterId(), got.GetClusterId(), "cluster_id at index %d", i)
					assert.Equal(t, want.GetState(), got.GetState(), "state for cluster %s", want.GetClusterId())
					assert.Equal(t, want.GetMessage(), got.GetMessage(), "message for cluster %s", want.GetClusterId())
				}
			}
		})
	}
}

func TestHealthCheckActor_Run(t *testing.T) {
	// Run() simply returns the input condition as-is (no changes).
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	mockBackend := backendsmocks.NewMockBackend(ctrl)
	registry := createHealthCheckTestRegistry(mockBackend)
	// No expectations set, backend should not be called

	actor := NewHealthCheckActor(nil, registry, zap.NewNop())

	resource := &v2pb.InferenceServer{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-server",
			Namespace: "test-namespace",
		},
		Spec: v2pb.InferenceServerSpec{
			BackendType: v2pb.BACKEND_TYPE_TRITON,
		},
	}

	// Provide an input condition with specific values
	condition := &apipb.Condition{
		Type:    "TritonHealthCheck",
		Status:  apipb.CONDITION_STATUS_FALSE,
		Reason:  "TestReason",
		Message: "TestMessage",
	}

	result, err := actor.Run(context.Background(), resource, condition)

	require.NoError(t, err)
	require.NotNil(t, result)
	// Run() returns the input condition as-is
	assert.Equal(t, apipb.CONDITION_STATUS_FALSE, result.Status)
	assert.Equal(t, "TestReason", result.Reason)
	assert.Equal(t, "TestMessage", result.Message)
	assert.Equal(t, "TritonHealthCheck", result.Type)
}
