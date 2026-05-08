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

	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/modelconfig/modelconfigmocks"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
)

func TestModelConfigProvisionActor_Retrieve(t *testing.T) {
	tests := []struct {
		name                string
		clusterTargets      []*v2pb.ClusterTarget
		clientFactoryErrors map[string]error
		setupMocks          func(*modelconfigmocks.MockModelConfigProvider)
		expectedStatus      apipb.ConditionStatus
		expectedMessage     string
		expectedReason      string
	}{
		{
			name:           "no clusters - vacuously provisioned",
			clusterTargets: nil,
			setupMocks:     func(_ *modelconfigmocks.MockModelConfigProvider) {},
			expectedStatus: apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:           "single cluster has model config",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mp *modelconfigmocks.MockModelConfigProvider) {
				mp.EXPECT().CheckModelConfigExists(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(true, nil)
			},
			expectedStatus: apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:           "single cluster missing model config",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mp *modelconfigmocks.MockModelConfigProvider) {
				mp.EXPECT().CheckModelConfigExists(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(false, nil)
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "ModelConfigNotFound",
			expectedReason:  "c1: model config not found",
		},
		{
			name:           "single cluster CheckModelConfigExists errors",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mp *modelconfigmocks.MockModelConfigProvider) {
				mp.EXPECT().CheckModelConfigExists(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(false, errors.New("api timeout"))
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "ModelConfigNotFound",
			expectedReason:  "c1: api timeout",
		},
		{
			name:                "GetClient errors for a single cluster",
			clusterTargets:      []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			clientFactoryErrors: map[string]error{"c1": errors.New("auth refused")},
			setupMocks:          func(_ *modelconfigmocks.MockModelConfigProvider) {},
			expectedStatus:      apipb.CONDITION_STATUS_FALSE,
			expectedMessage:     "ModelConfigNotFound",
			expectedReason:      "c1: client error: auth refused",
		},
		{
			name: "mixed: present + missing + check err + client err",
			clusterTargets: []*v2pb.ClusterTarget{
				{ClusterId: "c-ok"},
				{ClusterId: "c-missing"},
				{ClusterId: "c-err"},
				{ClusterId: "c-noclient"},
			},
			clientFactoryErrors: map[string]error{"c-noclient": errors.New("no token")},
			setupMocks: func(mp *modelconfigmocks.MockModelConfigProvider) {
				gomock.InOrder(
					mp.EXPECT().CheckModelConfigExists(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(true, nil),
					mp.EXPECT().CheckModelConfigExists(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(false, nil),
					mp.EXPECT().CheckModelConfigExists(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(false, errors.New("dial timeout")),
				)
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "ModelConfigNotFound",
			expectedReason:  "c-missing: model config not found; c-err: dial timeout; c-noclient: client error: no token",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mockProvider := modelconfigmocks.NewMockModelConfigProvider(ctrl)
			tt.setupMocks(mockProvider)

			factory := newClientFactoryDispatching(ctrl, tt.clientFactoryErrors)
			actor := NewModelConfigProvisionActor(factory, mockProvider, zap.NewNop())

			resource := &v2pb.InferenceServer{
				ObjectMeta: metav1.ObjectMeta{Name: "test-server", Namespace: "test-namespace"},
				Spec: v2pb.InferenceServerSpec{
					BackendType:    v2pb.BACKEND_TYPE_TRITON,
					ClusterTargets: tt.clusterTargets,
				},
			}
			condition := &apipb.Condition{Type: "ModelConfigProvision"}

			result, err := actor.Retrieve(context.Background(), resource, condition)

			require.NoError(t, err)
			assert.Equal(t, tt.expectedStatus, result.Status)
			assert.Equal(t, tt.expectedMessage, result.Message)
			assert.Equal(t, tt.expectedReason, result.Reason)
			assert.Equal(t, "ModelConfigProvision", result.Type)
		})
	}
}

func TestModelConfigProvisionActor_Run(t *testing.T) {
	tests := []struct {
		name                string
		clusterTargets      []*v2pb.ClusterTarget
		clientFactoryErrors map[string]error
		setupMocks          func(*modelconfigmocks.MockModelConfigProvider)
		expectedStatus      apipb.ConditionStatus
		expectedMessage     string
		expectedReason      string
	}{
		{
			name:           "all clusters already have model config - true",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}, {ClusterId: "c2"}},
			setupMocks: func(mp *modelconfigmocks.MockModelConfigProvider) {
				mp.EXPECT().CheckModelConfigExists(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(true, nil).Times(2)
			},
			expectedStatus: apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:           "first cluster needs work, CreateModelConfig succeeds - in progress",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}, {ClusterId: "c2"}},
			setupMocks: func(mp *modelconfigmocks.MockModelConfigProvider) {
				gomock.InOrder(
					mp.EXPECT().CheckModelConfigExists(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(false, nil),
					mp.EXPECT().CreateModelConfig(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace", gomock.Any(), gomock.Any()).
						Return(nil),
				)
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "RollingInProgress",
			expectedReason:  "provisioning cluster c1",
		},
		{
			name:           "first cluster needs work, CreateModelConfig fails",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mp *modelconfigmocks.MockModelConfigProvider) {
				gomock.InOrder(
					mp.EXPECT().CheckModelConfigExists(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
						Return(false, nil),
					mp.EXPECT().CreateModelConfig(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace", gomock.Any(), gomock.Any()).
						Return(errors.New("apply failed")),
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
			setupMocks:          func(_ *modelconfigmocks.MockModelConfigProvider) {},
			expectedStatus:      apipb.CONDITION_STATUS_FALSE,
			expectedMessage:     "ClientError",
			expectedReason:      "c1: auth refused",
		},
		{
			name:           "first cluster isDone errors",
			clusterTargets: []*v2pb.ClusterTarget{{ClusterId: "c1"}},
			setupMocks: func(mp *modelconfigmocks.MockModelConfigProvider) {
				mp.EXPECT().CheckModelConfigExists(gomock.Any(), gomock.Any(), gomock.Any(), "test-server", "test-namespace").
					Return(false, errors.New("api timeout"))
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "StatusCheckFailed",
			expectedReason:  "c1: api timeout",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mockProvider := modelconfigmocks.NewMockModelConfigProvider(ctrl)
			tt.setupMocks(mockProvider)

			factory := newClientFactoryDispatching(ctrl, tt.clientFactoryErrors)
			actor := NewModelConfigProvisionActor(factory, mockProvider, zap.NewNop())

			resource := &v2pb.InferenceServer{
				ObjectMeta: metav1.ObjectMeta{Name: "test-server", Namespace: "test-namespace"},
				Spec: v2pb.InferenceServerSpec{
					BackendType:    v2pb.BACKEND_TYPE_TRITON,
					ClusterTargets: tt.clusterTargets,
				},
			}
			condition := &apipb.Condition{Type: "ModelConfigProvision"}

			result, err := actor.Run(context.Background(), resource, condition)

			require.NoError(t, err)
			assert.Equal(t, tt.expectedStatus, result.Status)
			assert.Equal(t, tt.expectedMessage, result.Message)
			assert.Equal(t, tt.expectedReason, result.Reason)
			assert.Equal(t, "ModelConfigProvision", result.Type)
		})
	}
}
