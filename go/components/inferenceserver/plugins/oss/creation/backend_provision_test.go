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
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/backends"
	backendsmocks "github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/backends/backendsmocks"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/clientfactory/clientfactorymocks"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
)

// createTestRegistry creates a registry with the mock backend registered for Triton.
func createTestRegistry(mockBackend *backendsmocks.MockBackend) *backends.Registry {
	registry := backends.NewRegistry()
	registry.Register(v2pb.BACKEND_TYPE_TRITON, mockBackend)
	return registry
}

// newClientFactoryDispatching returns a MockClientFactory whose GetClient looks up the per-target
// error in clientErr. Targets without an entry get (nil, nil).
func newClientFactoryDispatching(ctrl *gomock.Controller, clientErr map[string]error) *clientfactorymocks.MockClientFactory {
	m := clientfactorymocks.NewMockClientFactory(ctrl)
	m.EXPECT().GetClient(gomock.Any(), gomock.Any()).DoAndReturn(
		func(_ context.Context, target *v2pb.ClusterTarget) (client.Client, error) {
			if err, ok := clientErr[target.GetClusterId()]; ok {
				return nil, err
			}
			return nil, nil
		},
	).AnyTimes()
	return m
}

func TestBackendProvisioningActor_Retrieve(t *testing.T) {
	tests := []struct {
		name                string
		clusterTargets      []*v2pb.ClusterTarget
		clientFactoryErrors map[string]error
		setupMocks          func(*backendsmocks.MockBackend)
		registryHasBackend  bool
		expectedStatus      apipb.ConditionStatus
		expectedMessage     string
		expectedReason      string
	}{
		{
			name:               "no clusters - vacuously provisioned",
			clusterTargets:     nil,
			setupMocks:         func(_ *backendsmocks.MockBackend) {},
			registryHasBackend: true,
			expectedStatus:     apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:           "single cluster serving",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mb *backendsmocks.MockBackend) {
				mb.EXPECT().GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(&backends.ServerStatus{State: v2pb.INFERENCE_SERVER_STATE_SERVING}, nil)
			},
			registryHasBackend: true,
			expectedStatus:     apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:           "single cluster still creating",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mb *backendsmocks.MockBackend) {
				mb.EXPECT().GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(&backends.ServerStatus{State: v2pb.INFERENCE_SERVER_STATE_CREATING}, nil)
			},
			registryHasBackend: true,
			expectedStatus:     apipb.CONDITION_STATUS_FALSE,
			expectedMessage:    "BackendProvisioningFailed",
			expectedReason:     "c1: state INFERENCE_SERVER_STATE_CREATING",
		},
		{
			name:           "single cluster GetServerStatus errors",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mb *backendsmocks.MockBackend) {
				mb.EXPECT().GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(nil, errors.New("api timeout"))
			},
			registryHasBackend: true,
			expectedStatus:     apipb.CONDITION_STATUS_FALSE,
			expectedMessage:    "BackendProvisioningFailed",
			expectedReason:     "c1: api timeout",
		},
		{
			name:                "GetClient errors for a single cluster",
			clusterTargets:      []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			clientFactoryErrors: map[string]error{"c1": errors.New("auth refused")},
			setupMocks:          func(_ *backendsmocks.MockBackend) {},
			registryHasBackend:  true,
			expectedStatus:      apipb.CONDITION_STATUS_FALSE,
			expectedMessage:     "BackendProvisioningFailed",
			expectedReason:      "c1: client error: auth refused",
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
			name: "mixed: serving + creating + status err + client err",
			clusterTargets: []*v2pb.ClusterTarget{
				{ClusterId: "c-ok"},
				{ClusterId: "c-creating"},
				{ClusterId: "c-err"},
				{ClusterId: "c-noclient"},
			},
			clientFactoryErrors: map[string]error{"c-noclient": errors.New("no token")},
			setupMocks: func(mb *backendsmocks.MockBackend) {
				gomock.InOrder(
					mb.EXPECT().GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(&backends.ServerStatus{State: v2pb.INFERENCE_SERVER_STATE_SERVING}, nil),
					mb.EXPECT().GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(&backends.ServerStatus{State: v2pb.INFERENCE_SERVER_STATE_CREATING}, nil),
					mb.EXPECT().GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(nil, errors.New("dial timeout")),
				)
			},
			registryHasBackend: true,
			expectedStatus:     apipb.CONDITION_STATUS_FALSE,
			expectedMessage:    "BackendProvisioningFailed",
			expectedReason:     "c-creating: state INFERENCE_SERVER_STATE_CREATING; c-err: dial timeout; c-noclient: client error: no token",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mockBackend := backendsmocks.NewMockBackend(ctrl)
			tt.setupMocks(mockBackend)

			registry := backends.NewRegistry()
			if tt.registryHasBackend {
				registry = createTestRegistry(mockBackend)
			}

			factory := newClientFactoryDispatching(ctrl, tt.clientFactoryErrors)
			actor := NewBackendProvisionActor(factory, registry, zap.NewNop())

			resource := &v2pb.InferenceServer{
				ObjectMeta: metav1.ObjectMeta{Name: "test-server", Namespace: "test-namespace"},
				Spec: v2pb.InferenceServerSpec{
					BackendType:    v2pb.BACKEND_TYPE_TRITON,
					ClusterTargets: tt.clusterTargets,
				},
			}
			condition := &apipb.Condition{Type: "BackendProvision"}

			result, err := actor.Retrieve(context.Background(), resource, condition)

			require.NoError(t, err)
			assert.Equal(t, tt.expectedStatus, result.Status)
			assert.Equal(t, tt.expectedMessage, result.Message)
			assert.Equal(t, tt.expectedReason, result.Reason)
			assert.Equal(t, "BackendProvision", result.Type)
		})
	}
}

func TestBackendProvisioningActor_Run(t *testing.T) {
	tests := []struct {
		name                string
		clusterTargets      []*v2pb.ClusterTarget
		clientFactoryErrors map[string]error
		setupMocks          func(*backendsmocks.MockBackend)
		registryHasBackend  bool
		expectedStatus      apipb.ConditionStatus
		expectedMessage     string
		expectedReason      string
	}{
		{
			name:               "all clusters already serving - true",
			clusterTargets:     []*v2pb.ClusterTarget{{ClusterId: "c1"}, {ClusterId: "c2"}},
			registryHasBackend: true,
			setupMocks: func(mb *backendsmocks.MockBackend) {
				mb.EXPECT().GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(&backends.ServerStatus{State: v2pb.INFERENCE_SERVER_STATE_SERVING}, nil).Times(2)
			},
			expectedStatus: apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:               "first cluster needs work, CreateServer succeeds - in progress",
			clusterTargets:     []*v2pb.ClusterTarget{{ClusterId: "c1"}, {ClusterId: "c2"}},
			registryHasBackend: true,
			setupMocks: func(mb *backendsmocks.MockBackend) {
				gomock.InOrder(
					mb.EXPECT().GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(&backends.ServerStatus{State: v2pb.INFERENCE_SERVER_STATE_CREATE_PENDING}, nil),
					mb.EXPECT().CreateServer(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).
						Return(&backends.ServerStatus{State: v2pb.INFERENCE_SERVER_STATE_CREATING}, nil),
				)
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "RollingInProgress",
			expectedReason:  "provisioning cluster c1",
		},
		{
			name:               "first cluster needs work, CreateServer fails",
			clusterTargets:     []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			registryHasBackend: true,
			setupMocks: func(mb *backendsmocks.MockBackend) {
				gomock.InOrder(
					mb.EXPECT().GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(&backends.ServerStatus{State: v2pb.INFERENCE_SERVER_STATE_CREATE_PENDING}, nil),
					mb.EXPECT().CreateServer(gomock.Any(), gomock.Any(), gomock.Any(), gomock.Any()).
						Return(nil, errors.New("apply failed")),
				)
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "ProvisionFailed",
			expectedReason:  "c1: apply failed",
		},
		{
			name:                "first cluster GetClient errors",
			clusterTargets:      []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			clientFactoryErrors: map[string]error{"c1": errors.New("auth refused")},
			registryHasBackend:  true,
			setupMocks:          func(_ *backendsmocks.MockBackend) {},
			expectedStatus:      apipb.CONDITION_STATUS_FALSE,
			expectedMessage:     "ClientError",
			expectedReason:      "c1: auth refused",
		},
		{
			name:               "first cluster isDone errors",
			clusterTargets:     []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			registryHasBackend: true,
			setupMocks: func(mb *backendsmocks.MockBackend) {
				mb.EXPECT().GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(nil, errors.New("api timeout"))
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "StatusCheckFailed",
			expectedReason:  "c1: api timeout",
		},
		{
			name:               "backend not in registry",
			clusterTargets:     []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			registryHasBackend: false,
			setupMocks:         func(_ *backendsmocks.MockBackend) {},
			expectedStatus:     apipb.CONDITION_STATUS_FALSE,
			expectedMessage:    "BackendNotFound",
			expectedReason:     "Failed to get backend: backend not found for type: BACKEND_TYPE_TRITON",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mockBackend := backendsmocks.NewMockBackend(ctrl)
			tt.setupMocks(mockBackend)

			registry := backends.NewRegistry()
			if tt.registryHasBackend {
				registry = createTestRegistry(mockBackend)
			}

			factory := newClientFactoryDispatching(ctrl, tt.clientFactoryErrors)
			actor := NewBackendProvisionActor(factory, registry, zap.NewNop())

			resource := &v2pb.InferenceServer{
				ObjectMeta: metav1.ObjectMeta{Name: "test-server", Namespace: "test-namespace"},
				Spec: v2pb.InferenceServerSpec{
					BackendType:    v2pb.BACKEND_TYPE_TRITON,
					ClusterTargets: tt.clusterTargets,
				},
			}
			condition := &apipb.Condition{Type: "BackendProvision"}

			result, err := actor.Run(context.Background(), resource, condition)

			require.NoError(t, err)
			assert.Equal(t, tt.expectedStatus, result.Status)
			assert.Equal(t, tt.expectedMessage, result.Message)
			assert.Equal(t, tt.expectedReason, result.Reason)
			assert.Equal(t, "BackendProvision", result.Type)
		})
	}
}
