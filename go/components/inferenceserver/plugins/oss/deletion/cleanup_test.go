package deletion

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
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/backends/backendsmocks"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/clientfactory/clientfactorymocks"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/modelconfig/modelconfigmocks"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
)

// createCleanupTestRegistry creates a registry with the mock backend registered for Triton.
func createCleanupTestRegistry(mockBackend *backendsmocks.MockBackend) *backends.Registry {
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

func TestCleanupActor_Retrieve(t *testing.T) {
	tests := []struct {
		name                string
		clusterTargets      []*v2pb.ClusterTarget
		clientFactoryErrors map[string]error
		setupMocks          func(*backendsmocks.MockBackend)
		expectedStatus      apipb.ConditionStatus
		expectedReason      string
		expectedMessage     string
		expectedErr         bool
	}{
		{
			name:           "no clusters - vacuously cleaned up",
			clusterTargets: nil,
			setupMocks:     func(_ *backendsmocks.MockBackend) {},
			expectedStatus: apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:           "inference server exists, cleanup not completed",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mockBackend *backendsmocks.MockBackend) {
				mockBackend.EXPECT().
					GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(&backends.ServerStatus{}, nil)
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedReason:  "c1: still present",
			expectedMessage: "CleanupInProgress",
		},
		{
			name:           "inference server does not exist, cleanup completed",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mockBackend *backendsmocks.MockBackend) {
				mockBackend.EXPECT().
					GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(nil, errors.New("inference server not found"))
			},
			expectedStatus: apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:                "GetClient errors for a single cluster",
			clusterTargets:      []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			clientFactoryErrors: map[string]error{"c1": errors.New("auth refused")},
			setupMocks:          func(_ *backendsmocks.MockBackend) {},
			expectedStatus:      apipb.CONDITION_STATUS_FALSE,
			expectedReason:      "c1: client error: auth refused",
			expectedMessage:     "CleanupInProgress",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mockBackend := backendsmocks.NewMockBackend(ctrl)
			mockConfigMapProvider := modelconfigmocks.NewMockModelConfigProvider(ctrl)
			registry := createCleanupTestRegistry(mockBackend)

			tt.setupMocks(mockBackend)

			factory := newClientFactoryDispatching(ctrl, tt.clientFactoryErrors)
			actor := NewCleanupActor(factory, registry, mockConfigMapProvider, zap.NewNop())

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
				Type: "TritonCleanup",
			}

			result, err := actor.Retrieve(context.Background(), resource, condition)

			if tt.expectedErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tt.expectedStatus, result.Status)
				assert.Equal(t, tt.expectedReason, result.Reason)
				assert.Equal(t, tt.expectedMessage, result.Message)
				assert.Equal(t, "TritonCleanup", result.Type)
			}
		})
	}
}

func TestCleanupActor_Run(t *testing.T) {
	tests := []struct {
		name                string
		clusterTargets      []*v2pb.ClusterTarget
		clientFactoryErrors map[string]error
		setupMocks          func(*backendsmocks.MockBackend, *modelconfigmocks.MockModelConfigProvider)
		expectedStatus      apipb.ConditionStatus
		expectedReason      string
		expectedMessage     string
	}{
		{
			name:           "all clusters already cleaned up - true",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}, {ClusterId: "c2"}},
			setupMocks: func(mockBackend *backendsmocks.MockBackend, _ *modelconfigmocks.MockModelConfigProvider) {
				// isDone returns true for both clusters (GetServerStatus errors → server gone).
				mockBackend.EXPECT().
					GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(nil, errors.New("not found")).Times(2)
			},
			expectedStatus: apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:           "successful cleanup, all resources deleted - in progress",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}, {ClusterId: "c2"}},
			setupMocks: func(mockBackend *backendsmocks.MockBackend, mockConfigMap *modelconfigmocks.MockModelConfigProvider) {
				gomock.InOrder(
					// isDone for c1: server still present.
					mockBackend.EXPECT().
						GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(&backends.ServerStatus{}, nil),
					// doWork for c1: delete model config, then delete server.
					mockConfigMap.EXPECT().
						DeleteModelConfig(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(nil),
					mockBackend.EXPECT().
						DeleteServer(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(nil),
				)
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedReason:  "provisioning cluster c1",
			expectedMessage: "RollingInProgress",
		},
		{
			name:           "configmap deletion fails, cleanup continues",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mockBackend *backendsmocks.MockBackend, mockConfigMap *modelconfigmocks.MockModelConfigProvider) {
				gomock.InOrder(
					mockBackend.EXPECT().
						GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(&backends.ServerStatus{}, nil),
					// ConfigMap deletion fails but is logged, not returned.
					mockConfigMap.EXPECT().
						DeleteModelConfig(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(errors.New("configmap not found")),
					// Inference server deletion still runs and succeeds.
					mockBackend.EXPECT().
						DeleteServer(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(nil),
				)
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedReason:  "provisioning cluster c1",
			expectedMessage: "RollingInProgress",
		},
		{
			name:           "inference server deletion fails, returns ProvisionFailed",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mockBackend *backendsmocks.MockBackend, mockConfigMap *modelconfigmocks.MockModelConfigProvider) {
				gomock.InOrder(
					mockBackend.EXPECT().
						GetServerStatus(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(&backends.ServerStatus{}, nil),
					mockConfigMap.EXPECT().
						DeleteModelConfig(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(nil),
					mockBackend.EXPECT().
						DeleteServer(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(errors.New("failed to delete deployment")),
				)
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedReason:  "c1: failed to delete deployment",
			expectedMessage: "ProvisionFailed",
		},
		{
			name:                "GetClient errors for a single cluster",
			clusterTargets:      []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			clientFactoryErrors: map[string]error{"c1": errors.New("auth refused")},
			setupMocks:          func(_ *backendsmocks.MockBackend, _ *modelconfigmocks.MockModelConfigProvider) {},
			expectedStatus:      apipb.CONDITION_STATUS_FALSE,
			expectedReason:      "c1: auth refused",
			expectedMessage:     "ClientError",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mockBackend := backendsmocks.NewMockBackend(ctrl)
			mockConfigMapProvider := modelconfigmocks.NewMockModelConfigProvider(ctrl)
			registry := createCleanupTestRegistry(mockBackend)

			tt.setupMocks(mockBackend, mockConfigMapProvider)

			factory := newClientFactoryDispatching(ctrl, tt.clientFactoryErrors)
			actor := NewCleanupActor(factory, registry, mockConfigMapProvider, zap.NewNop())

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
				Type: "TritonCleanup",
			}

			result, err := actor.Run(context.Background(), resource, condition)

			require.NoError(t, err)
			require.NotNil(t, result)
			assert.Equal(t, tt.expectedStatus, result.Status)
			assert.Equal(t, tt.expectedReason, result.Reason)
			assert.Equal(t, tt.expectedMessage, result.Message)
		})
	}
}
