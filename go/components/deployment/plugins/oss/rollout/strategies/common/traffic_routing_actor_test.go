package common

import (
	"context"
	"errors"
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	"k8s.io/client-go/dynamic"

	"github.com/michelangelo-ai/michelangelo/go/components/deployment/route/routemocks"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/backends"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/clientfactory/clientfactorymocks"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/modelconfig/modelconfigmocks"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
)

// trafficMocks groups the mocks used by traffic-routing tests. Distinct from rolloutMocks
// because this actor uses the dynamic client + route provider rather than the typed client +
// backend / model config provider.
type trafficMocks struct {
	factory       *clientfactorymocks.MockClientFactory
	routeProvider *routemocks.MockRouteProvider
}

// newTrafficFixture builds a Params + target wired to the supplied mocks. dynamicClientErr
// lets a test inject a GetDynamicClient failure without re-mocking the factory each time.
func newTrafficFixture(t *testing.T, dynamicClientErr error) (Params, *v2pb.ClusterTarget, *trafficMocks) {
	t.Helper()
	ctrl := gomock.NewController(t)
	t.Cleanup(ctrl.Finish)

	mocks := &trafficMocks{
		factory:       clientfactorymocks.NewMockClientFactory(ctrl),
		routeProvider: routemocks.NewMockRouteProvider(ctrl),
	}

	mocks.factory.EXPECT().GetDynamicClient(gomock.Any(), gomock.Any()).
		Return(dynamic.Interface(nil), dynamicClientErr).AnyTimes()

	params := Params{
		ClientFactory:       mocks.factory,
		BackendRegistry:     backends.NewRegistry(),
		ModelConfigProvider: modelconfigmocks.NewMockModelConfigProvider(ctrl),
		RouteProvider:       mocks.routeProvider,
		Logger:              zap.NewNop(),
	}
	target := &v2pb.ClusterTarget{ClusterId: testCluster}
	return params, target, mocks
}

func TestTrafficRoutingActor_Retrieve(t *testing.T) {
	tests := []struct {
		name              string
		dynamicClientErr  error
		setupMocks        func(*trafficMocks)
		expectedStatus    apipb.ConditionStatus
		expectedReasonSub string
	}{
		{
			name:              "GetDynamicClient errors",
			dynamicClientErr:  errors.New("dial timeout"),
			setupMocks:        func(*trafficMocks) {},
			expectedStatus:    apipb.CONDITION_STATUS_FALSE,
			expectedReasonSub: "dial timeout",
		},
		{
			name: "check deployment route status fails",
			setupMocks: func(m *trafficMocks) {
				m.routeProvider.EXPECT().CheckDeploymentRouteStatus(gomock.Any(), gomock.Any(), gomock.Any(),
					testDeploymentName, testNamespace, testISName, testModelName).
					Return(false, errors.New("api error"))
			},
			expectedStatus:    apipb.CONDITION_STATUS_FALSE,
			expectedReasonSub: "api error",
		},
		{
			name: "deployment route not configured",
			setupMocks: func(m *trafficMocks) {
				m.routeProvider.EXPECT().CheckDeploymentRouteStatus(gomock.Any(), gomock.Any(), gomock.Any(),
					testDeploymentName, testNamespace, testISName, testModelName).
					Return(false, nil)
			},
			expectedStatus:    apipb.CONDITION_STATUS_FALSE,
			expectedReasonSub: "HTTPRoute in cluster c1 not pointing at model model-v1",
		},
		{
			name: "traffic routing configured successfully",
			setupMocks: func(m *trafficMocks) {
				m.routeProvider.EXPECT().CheckDeploymentRouteStatus(gomock.Any(), gomock.Any(), gomock.Any(),
					testDeploymentName, testNamespace, testISName, testModelName).
					Return(true, nil)
			},
			expectedStatus: apipb.CONDITION_STATUS_TRUE,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params, target, mocks := newTrafficFixture(t, tt.dynamicClientErr)
			tt.setupMocks(mocks)

			actor := NewTrafficRoutingActor(params, target)
			got, err := actor.Retrieve(context.Background(), rolloutDeployment(""), &apipb.Condition{})

			require.NoError(t, err)
			assert.Equal(t, tt.expectedStatus, got.Status)
			if tt.expectedReasonSub != "" {
				assert.Contains(t, got.Reason, tt.expectedReasonSub)
			}
		})
	}
}

func TestTrafficRoutingActor_Run(t *testing.T) {
	tests := []struct {
		name              string
		dynamicClientErr  error
		setupMocks        func(*trafficMocks)
		expectedStatus    apipb.ConditionStatus
		expectedReasonSub string
	}{
		{
			name:              "GetDynamicClient errors",
			dynamicClientErr:  errors.New("dial timeout"),
			setupMocks:        func(*trafficMocks) {},
			expectedStatus:    apipb.CONDITION_STATUS_FALSE,
			expectedReasonSub: "dial timeout",
		},
		{
			name: "add deployment route fails",
			setupMocks: func(m *trafficMocks) {
				m.routeProvider.EXPECT().EnsureDeploymentRoute(gomock.Any(), gomock.Any(), gomock.Any(),
					testDeploymentName, testNamespace, testISName, testModelName).
					Return(errors.New("route creation failed"))
			},
			expectedStatus:    apipb.CONDITION_STATUS_FALSE,
			expectedReasonSub: "route creation failed",
		},
		{
			name: "traffic routing configured successfully",
			setupMocks: func(m *trafficMocks) {
				m.routeProvider.EXPECT().EnsureDeploymentRoute(gomock.Any(), gomock.Any(), gomock.Any(),
					testDeploymentName, testNamespace, testISName, testModelName).
					Return(nil)
			},
			expectedStatus: apipb.CONDITION_STATUS_TRUE,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			params, target, mocks := newTrafficFixture(t, tt.dynamicClientErr)
			tt.setupMocks(mocks)

			actor := NewTrafficRoutingActor(params, target)
			got, err := actor.Run(context.Background(), rolloutDeployment(""), &apipb.Condition{})

			require.NoError(t, err)
			assert.Equal(t, tt.expectedStatus, got.Status)
			if tt.expectedReasonSub != "" {
				assert.Contains(t, got.Reason, tt.expectedReasonSub)
			}
		})
	}
}

func TestTrafficRoutingActor_GetType(t *testing.T) {
	params, target, _ := newTrafficFixture(t, nil)
	actor := NewTrafficRoutingActor(params, target)
	assert.Equal(t, "TrafficRoutingConfigured-"+testCluster, actor.GetType())
}
