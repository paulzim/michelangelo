package provider

import (
	"context"
	"errors"
	"testing"

	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/zap"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	maconfig "github.com/michelangelo-ai/michelangelo/go/base/config"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/clientfactory/clientfactorymocks"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/endpoints"
	v2pb "github.com/michelangelo-ai/michelangelo/proto-go/api/v2"
)

const (
	testGatewayName      = "ma-gateway-istio"
	testGatewayNamespace = "default"
	testPortName         = "http"
	testClusterID        = "c1"
)

func newTestConfig() maconfig.InferenceServerConfig {
	return maconfig.InferenceServerConfig{
		Gateway: maconfig.GatewayConfig{
			ServiceName:      testGatewayName,
			ServiceNamespace: testGatewayNamespace,
			PortName:         testPortName,
		},
	}
}

func newScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(scheme))
	return scheme
}

// gatewayService builds the gateway Service with the given port name and NodePort.
// Pass portName="" to omit the matching named port entirely.
func gatewayService(portName string, nodePort int32) *corev1.Service {
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      testGatewayName,
			Namespace: testGatewayNamespace,
		},
		Spec: corev1.ServiceSpec{
			Type: corev1.ServiceTypeNodePort,
		},
	}
	if portName != "" {
		svc.Spec.Ports = []corev1.ServicePort{{
			Name:     portName,
			Port:     80,
			NodePort: nodePort,
		}}
	}
	return svc
}

// nodeWithAddresses builds a Node with the given address types/values. Use
// corev1.NodeInternalIP / corev1.NodeExternalIP as types in pairs.
func nodeWithAddresses(name string, addrs ...corev1.NodeAddress) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Status:     corev1.NodeStatus{Addresses: addrs},
	}
}

// newK8sFixture wires the MockClientFactory to return the supplied fake k8s
// client (or clientErr) for testClusterID.
func newK8sFixture(
	t *testing.T,
	objects []client.Object,
	clientErr error,
) (*K8sProvider, *v2pb.ClusterTarget) {
	t.Helper()
	ctrl := gomock.NewController(t)
	t.Cleanup(ctrl.Finish)

	scheme := newScheme(t)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(objects...).Build()

	mockFactory := clientfactorymocks.NewMockClientFactory(ctrl)
	mockFactory.EXPECT().GetClient(gomock.Any(), gomock.Any()).DoAndReturn(
		func(_ context.Context, target *v2pb.ClusterTarget) (client.Client, error) {
			if clientErr != nil {
				return nil, clientErr
			}
			return fakeClient, nil
		},
	).AnyTimes()

	provider := NewK8sProvider(mockFactory, newTestConfig(), zap.NewNop())
	target := &v2pb.ClusterTarget{ClusterId: testClusterID}
	return provider, target
}

func TestK8sProvider_Resolve(t *testing.T) {
	tests := []struct {
		name        string
		objects     []client.Object
		clientErr   error
		expectedEP  endpoints.Endpoint
		expectedErr string // substring match; "" means no error expected
	}{
		{
			name: "happy path",
			objects: []client.Object{
				gatewayService(testPortName, 31234),
				nodeWithAddresses("node-0", corev1.NodeAddress{
					Type: corev1.NodeInternalIP, Address: "10.0.0.5",
				}),
			},
			expectedEP: endpoints.Endpoint{Host: "10.0.0.5", Port: 31234, Scheme: "http"},
		},
		{
			name:        "GetClient errors",
			clientErr:   errors.New("auth refused"),
			expectedErr: `get client for cluster "c1"`,
		},
		{
			name:        "gateway service missing",
			objects:     nil,
			expectedErr: "get gateway service default/ma-gateway-istio",
		},
		{
			name: "port name mismatch",
			objects: []client.Object{
				gatewayService("grpc", 31234),
				nodeWithAddresses("node-0", corev1.NodeAddress{
					Type: corev1.NodeInternalIP, Address: "10.0.0.5",
				}),
			},
			expectedErr: `has no port named "http"`,
		},
		{
			name: "named port has no NodePort",
			objects: []client.Object{
				gatewayService(testPortName, 0),
				nodeWithAddresses("node-0", corev1.NodeAddress{
					Type: corev1.NodeInternalIP, Address: "10.0.0.5",
				}),
			},
			expectedErr: `has no NodePort on port "http"`,
		},
		{
			name: "no nodes",
			objects: []client.Object{
				gatewayService(testPortName, 31234),
			},
			expectedErr: `no node on cluster "c1" reported an InternalIP`,
		},
		{
			name: "no node has InternalIP",
			objects: []client.Object{
				gatewayService(testPortName, 31234),
				nodeWithAddresses("node-0", corev1.NodeAddress{
					Type: corev1.NodeExternalIP, Address: "1.2.3.4",
				}),
			},
			expectedErr: `no node on cluster "c1" reported an InternalIP`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			provider, target := newK8sFixture(t, tt.objects, tt.clientErr)

			ep, err := provider.Resolve(context.Background(), target)

			if tt.expectedErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.expectedErr)
				assert.Equal(t, endpoints.Endpoint{}, ep)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tt.expectedEP, ep)
		})
	}
}
