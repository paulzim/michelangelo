package endpoints

import (
	"go.uber.org/config"
	"go.uber.org/fx"
	"sigs.k8s.io/controller-runtime/pkg/client"

	maconfig "github.com/michelangelo-ai/michelangelo/go/base/config"
)

// Module wires the Publisher with the in-cluster Kubernetes client and
// the InferenceServerConfig from the typed config provider. The Provider is
// provided separately by an environment-specific module (for example,
// provider.Module for k8s clusters).
var Module = fx.Options(
	fx.Provide(newDefaultPublisher),
	fx.Provide(newInferenceServerConfig),
)

func newDefaultPublisher(kubeClient client.Client) Publisher {
	return NewDefaultPublisher(kubeClient, kubeClient.Scheme())
}

func newInferenceServerConfig(provider config.Provider) (maconfig.InferenceServerConfig, error) {
	return maconfig.GetInferenceServerConfig(provider)
}
