package provider

import (
	"go.uber.org/fx"
	"go.uber.org/zap"

	maconfig "github.com/michelangelo-ai/michelangelo/go/base/config"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/clientfactory"
	"github.com/michelangelo-ai/michelangelo/go/components/inferenceserver/endpoints"
)

// Module binds the k8s Provider into the fx graph. Include this module
// alongside endpoints.Module when running against k8s clusters.
var Module = fx.Options(
	fx.Provide(newK8sProvider),
)

func newK8sProvider(clientFactory clientfactory.ClientFactory, isConfig maconfig.InferenceServerConfig, logger *zap.Logger) endpoints.Provider {
	return NewK8sProvider(clientFactory, isConfig, logger)
}
