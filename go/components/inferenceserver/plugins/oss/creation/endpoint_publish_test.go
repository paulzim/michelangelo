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

	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/endpoints"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/endpoints/endpointsmocks"
	apipb "github.com/michelangelo-ai/michelangelo/proto-go/api"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
)

// makeIS builds a minimal InferenceServer with the given cluster IDs in spec.ClusterTargets.
func makeIS(clusterIDs ...string) *v2pb.InferenceServer {
	targets := make([]*v2pb.ClusterTarget, 0, len(clusterIDs))
	for _, id := range clusterIDs {
		targets = append(targets, &v2pb.ClusterTarget{ClusterId: id})
	}
	return &v2pb.InferenceServer{
		ObjectMeta: metav1.ObjectMeta{Name: "test-server", Namespace: "test-namespace"},
		Spec:       v2pb.InferenceServerSpec{ClusterTargets: targets},
	}
}

func TestEndpointPublishActor_Retrieve(t *testing.T) {
	tests := []struct {
		name            string
		clusterIDs      []string
		published       map[string]endpoints.Endpoint
		getErr          error
		expectedStatus  apipb.ConditionStatus
		expectedMessage string
		expectedReason  string
	}{
		{
			name:           "no clusters; vacuously true",
			clusterIDs:     nil,
			published:      map[string]endpoints.Endpoint{},
			expectedStatus: apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:       "all clusters published",
			clusterIDs: []string{"c1", "c2"},
			published: map[string]endpoints.Endpoint{
				"c1": {Host: "10.0.0.1", Port: 31001, Scheme: "http"},
				"c2": {Host: "10.0.0.2", Port: 31002, Scheme: "http"},
			},
			expectedStatus: apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:       "single cluster missing from published",
			clusterIDs: []string{"c1", "c2"},
			published: map[string]endpoints.Endpoint{
				"c1": {Host: "10.0.0.1", Port: 31001, Scheme: "http"},
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "EndpointSliceMissing",
			expectedReason:  "c2",
		},
		{
			name:       "multiple missing clusters",
			clusterIDs: []string{"c1", "c2", "c3"},
			published: map[string]endpoints.Endpoint{
				"c2": {Host: "10.0.0.2", Port: 31002, Scheme: "http"},
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "EndpointSliceMissing",
			expectedReason:  "c1,c3",
		},
		{
			name:       "orphan endpoint slice",
			clusterIDs: []string{"c1"},
			published: map[string]endpoints.Endpoint{
				"c1":     {Host: "10.0.0.1", Port: 31001, Scheme: "http"},
				"c2-old": {Host: "10.0.0.2", Port: 31002, Scheme: "http"},
			},
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "OrphanEndpointSlice",
			expectedReason:  "c2-old",
		},
		{
			name:            "publisher.Get errors",
			clusterIDs:      []string{"c1"},
			getErr:          errors.New("boom"),
			expectedStatus:  apipb.CONDITION_STATUS_FALSE,
			expectedMessage: "GetFailed",
			expectedReason:  "boom",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mockPublisher := endpointsmocks.NewMockPublisher(ctrl)
			mockProvider := endpointsmocks.NewMockProvider(ctrl)

			mockPublisher.EXPECT().Get(gomock.Any(), gomock.Any()).Return(tt.published, tt.getErr)

			actor := NewEndpointPublishActor(mockPublisher, mockProvider, zap.NewNop())

			condition := &apipb.Condition{Type: "EndpointPublish"}
			result, err := actor.Retrieve(context.Background(), makeIS(tt.clusterIDs...), condition)

			require.NoError(t, err)
			assert.Equal(t, tt.expectedStatus, result.Status)
			assert.Equal(t, tt.expectedMessage, result.Message)
			assert.Equal(t, tt.expectedReason, result.Reason)
			assert.Equal(t, "EndpointPublish", result.Type)
		})
	}
}

func TestEndpointPublishActor_Run(t *testing.T) {
	ep1 := endpoints.Endpoint{Host: "10.0.0.1", Port: 31001, Scheme: "http"}
	ep2 := endpoints.Endpoint{Host: "10.0.0.2", Port: 31002, Scheme: "http"}

	tests := []struct {
		name              string
		clusterIDs        []string
		resolveResults    map[string]endpoints.Endpoint // happy resolves
		resolveErrors     map[string]error              // per-cluster resolve errors
		syncErr           error
		expectSyncWithMap map[string]endpoints.Endpoint
		expectedStatus    apipb.ConditionStatus
		expectedMessage   string
		expectedReasons   []string // substring matches (reason order undefined for partial failures)
	}{
		{
			name:              "no clusters; sync called with empty map",
			clusterIDs:        nil,
			resolveResults:    nil,
			expectSyncWithMap: map[string]endpoints.Endpoint{},
			expectedStatus:    apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:              "all resolves succeed",
			clusterIDs:        []string{"c1", "c2"},
			resolveResults:    map[string]endpoints.Endpoint{"c1": ep1, "c2": ep2},
			expectSyncWithMap: map[string]endpoints.Endpoint{"c1": ep1, "c2": ep2},
			expectedStatus:    apipb.CONDITION_STATUS_TRUE,
		},
		{
			name:              "one cluster resolve errors; partial publish",
			clusterIDs:        []string{"c1", "c2"},
			resolveResults:    map[string]endpoints.Endpoint{"c1": ep1},
			resolveErrors:     map[string]error{"c2": errors.New("dial timeout")},
			expectSyncWithMap: map[string]endpoints.Endpoint{"c1": ep1},
			expectedStatus:    apipb.CONDITION_STATUS_UNKNOWN,
			expectedMessage:   "PartialEndpointPublish",
			expectedReasons:   []string{"c2: resolve: dial timeout"},
		},
		{
			name:       "all clusters resolve err; sync still called with empty map",
			clusterIDs: []string{"c1", "c2"},
			resolveErrors: map[string]error{
				"c1": errors.New("dial timeout"),
				"c2": errors.New("auth refused"),
			},
			expectSyncWithMap: map[string]endpoints.Endpoint{},
			expectedStatus:    apipb.CONDITION_STATUS_UNKNOWN,
			expectedMessage:   "PartialEndpointPublish",
			expectedReasons:   []string{"c1: resolve: dial timeout", "c2: resolve: auth refused"},
		},
		{
			name:              "publisher.Sync errors",
			clusterIDs:        []string{"c1"},
			resolveResults:    map[string]endpoints.Endpoint{"c1": ep1},
			syncErr:           errors.New("apply failed"),
			expectSyncWithMap: map[string]endpoints.Endpoint{"c1": ep1},
			expectedStatus:    apipb.CONDITION_STATUS_FALSE,
			expectedMessage:   "SyncFailed",
			expectedReasons:   []string{"apply failed"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctrl := gomock.NewController(t)
			defer ctrl.Finish()

			mockPublisher := endpointsmocks.NewMockPublisher(ctrl)
			mockProvider := endpointsmocks.NewMockProvider(ctrl)

			mockProvider.EXPECT().Resolve(gomock.Any(), gomock.Any()).DoAndReturn(
				func(_ context.Context, target *v2pb.ClusterTarget) (endpoints.Endpoint, error) {
					id := target.GetClusterId()
					if err, ok := tt.resolveErrors[id]; ok {
						return endpoints.Endpoint{}, err
					}
					return tt.resolveResults[id], nil
				},
			).Times(len(tt.clusterIDs))

			// Capture the actual map passed to Sync so we can assert it matches the
			// resolved set (catches drift between resolve outputs and what gets synced).
			var gotSyncMap map[string]endpoints.Endpoint
			mockPublisher.EXPECT().Sync(gomock.Any(), gomock.Any(), gomock.Any()).DoAndReturn(
				func(_ context.Context, _ *v2pb.InferenceServer, m map[string]endpoints.Endpoint) error {
					gotSyncMap = m
					return tt.syncErr
				},
			)

			actor := NewEndpointPublishActor(mockPublisher, mockProvider, zap.NewNop())

			condition := &apipb.Condition{Type: "EndpointPublish"}
			result, err := actor.Run(context.Background(), makeIS(tt.clusterIDs...), condition)

			require.NoError(t, err)
			assert.Equal(t, tt.expectSyncWithMap, gotSyncMap, "Sync called with wrong endpoint map")
			assert.Equal(t, tt.expectedStatus, result.Status)
			assert.Equal(t, tt.expectedMessage, result.Message)
			for _, want := range tt.expectedReasons {
				assert.Contains(t, result.Reason, want)
			}
			assert.Equal(t, "EndpointPublish", result.Type)
		})
	}
}
